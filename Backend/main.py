from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np
import tensorflow as tf
import cv2
from werkzeug.utils import secure_filename
from lime import lime_image
from skimage.segmentation import mark_boundaries
import base64
import matplotlib.cm as cm
from concurrent.futures import ThreadPoolExecutor # Import ThreadPoolExecutor

# --- (Keep Flask app setup, CORS, Paths, Constants, Model Loading, Helpers as before) ---

app = Flask(__name__)
CORS(app)

# Paths
MRI_MODEL_PATH_DENSENET = "D:/SAI/MRI_DenseNet201_CBAM_best.h5"
MRI_MODEL_PATH_RESNET = "D:/SAI/MRI_ResNet50_CBAM_E.h5"
PET_MODEL_PATH = "D:/SAI/PET_VGG19_CBAM_Enhanced.h5"
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Constants
IMG_SIZE = 128
CLASSES = ['AD', 'CN', 'EMCI', 'LMCI']

# Load Models
print("Loading models...")
mri_densenet = tf.keras.models.load_model(MRI_MODEL_PATH_DENSENET)
mri_resnet = tf.keras.models.load_model(MRI_MODEL_PATH_RESNET)
pet_model = tf.keras.models.load_model(PET_MODEL_PATH)
print("Models loaded.")

#Validation MRI
# Load Validator Models
MRI_VALIDATOR_PATH = "D:/SAI/mri_validator_model.h5"
PET_VALIDATOR_PATH = "D:/SAI/pet_validator_model.h5"

print("Loading validator models...")
mri_validator = tf.keras.models.load_model(MRI_VALIDATOR_PATH)
pet_validator = tf.keras.models.load_model(PET_VALIDATOR_PATH)
print("Validator models loaded.")

def validate_image_with_model(image_path, validator_model):
    try:
        img = cv2.imread(image_path)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = validator_model.predict(img)[0][0]
        return prediction > 0.5  # True = Valid Image
    except Exception as e:
        print(f"Validation Error: {str(e)}")
        return False


# --- Helper Functions (encode_image_to_base64, apply_matplotlib_colormap, preprocess_image, get_brain_mask) ---

def encode_image_to_base64(image):
    # Ensure image is uint8 before encoding
    if image.dtype != np.uint8:
        # Handle potential float images from LIME or overlays
        if image.max() <= 1.0 and image.min() >= 0.0:
             image = (image * 255).astype(np.uint8)
        else:
            # Attempt safe conversion if values might be outside 0-1 or 0-255
            image = np.clip(image, 0, 255).astype(np.uint8)

    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')

def apply_matplotlib_colormap(heatmap):
    cmap = cm.get_cmap('jet')
    # Normalize heatmap to 0-1 range before applying colormap
    heatmap_normalized = np.clip(heatmap, 0, 1)
    colored_heatmap = cmap(heatmap_normalized)[:, :, :3] # Take only RGB
    return (colored_heatmap * 255).astype(np.uint8)

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")
    # Ensure input is 3 channel BGR
    if len(img.shape) == 2: # Grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4: # RGBA
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_normalized = img_resized.astype(np.float32) / 255.0
    return np.expand_dims(img_normalized, axis=0), img_normalized # Return input tensor and raw-like (resized, normalized)

def get_brain_mask(image, threshold=0.1):
    if len(image.shape) == 3 and image.shape[2] == 3:
         gray_image = np.mean(image, axis=-1)
    elif len(image.shape) == 2:
         gray_image = image
    else:
         # Fallback or raise error if unexpected shape
         gray_image = image[:,:,0] # Assume first channel if unsure

    mask = (gray_image > threshold).astype(np.float32)
    return mask

# --- Explanation Functions (with Tuned Parameters) ---

def integrated_gradients(image, model, class_index, steps=50): # Reduced steps
    image = image.astype(np.float32)
    baseline = np.zeros_like(image).astype(np.float32)
    interpolated_images = [baseline + (float(i) / steps) * (image - baseline) for i in range(steps + 1)]
    interpolated_images = tf.convert_to_tensor(np.array(interpolated_images), dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(interpolated_images)
        preds = model(interpolated_images)
        target = preds[:, class_index]

    grads = tape.gradient(target, interpolated_images)
    if grads is None:
         print(f"Warning: IG Gradients are None for class {class_index}. Returning zeros.")
         return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

    grads = grads.numpy()
    avg_grads = np.mean(grads, axis=0)
    integrated_grads = (image - baseline) * avg_grads
    attributions = np.mean(np.abs(integrated_grads), axis=-1)
    attributions = np.nan_to_num(attributions) # Handle potential NaNs

    # Normalize 0-1
    min_val, max_val = np.min(attributions), np.max(attributions)
    if max_val > min_val:
        attributions = (attributions - min_val) / (max_val - min_val + 1e-8)
    else:
        attributions = np.zeros_like(attributions) # Avoid division by zero if flat

    # Mask out background AFTER normalization
    brain_mask = get_brain_mask(image)
    return attributions * brain_mask


def grad_cam(image, model, class_idx, layer_name):
    image_tensor = tf.convert_to_tensor(image[np.newaxis, ...], dtype=tf.float32)

    # Find the layer if name is partial or needs checking
    try:
        target_layer = model.get_layer(layer_name)
    except ValueError:
        print(f"Warning: Layer '{layer_name}' not found directly. Searching...")
        found = False
        for layer in reversed(model.layers): # Search backwards (likely deeper)
             if layer_name in layer.name and isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Add, tf.keras.layers.Concatenate)): # Common layer types
                  target_layer = layer
                  print(f"Using layer: {target_layer.name}")
                  found = True
                  break
        if not found:
             print(f"Error: Could not find suitable layer containing '{layer_name}'. Using last conv layer.")
             # Fallback: try to find the last convolutional layer
             for layer in reversed(model.layers):
                  if isinstance(layer, tf.keras.layers.Conv2D):
                       target_layer = layer
                       print(f"Fallback: Using last Conv2D layer: {target_layer.name}")
                       break
             if target_layer is None: # Should not happen if model has Conv layers
                 raise ValueError("Cannot find any Conv layer for GradCAM")

    grad_model = tf.keras.models.Model([model.inputs], [target_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_tensor)
        if class_idx >= predictions.shape[1]:
             print(f"Warning: class_idx {class_idx} out of bounds for prediction shape {predictions.shape}. Using index 0.")
             class_idx = 0
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
         print(f"Warning: GradCAM Gradients are None for layer {target_layer.name}, class {class_idx}. Returning zeros.")
         return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

    # Handle potential batch dimension in grads if input was batched differently
    if len(grads.shape) == 4: # (batch, h, w, c) - expected
        pooled_grads = tf.reduce_mean(grads, axis=(1, 2)) # Pool over spatial dims H, W
    elif len(grads.shape) == 3: # Might happen in some architectures (h, w, c)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1)) # Pool over spatial dims H, W
        pooled_grads = tf.expand_dims(pooled_grads, axis=0) # Add back batch dim
    else:
        print(f"Warning: Unexpected grads shape in GradCAM: {grads.shape}. Returning zeros.")
        return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

    # Ensure conv_outputs also has batch dim
    if len(conv_outputs.shape) == 3:
         conv_outputs = tf.expand_dims(conv_outputs, axis=0)

    # Element-wise multiply pooled grads with feature map channels
    heatmap = tf.multiply(pooled_grads[..., tf.newaxis, tf.newaxis, :], conv_outputs)
    heatmap = tf.reduce_sum(heatmap, axis=-1).numpy().squeeze() # Sum over channels
    heatmap = np.maximum(heatmap, 0) # ReLU

    # Normalize 0-1
    max_val = np.max(heatmap)
    if max_val > 0:
        heatmap /= max_val

    return cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE)) # Resize to original image size


def lime_explainer(image, model):
    explainer = lime_image.LimeImageExplainer()
    # Use a lambda to ensure predict function has correct output shape for LIME
    def predict_fn_for_lime(images):
        return model.predict(images)

    explanation = explainer.explain_instance(
        image.astype('double'),
        predict_fn_for_lime, # Use the wrapper
        top_labels=len(CLASSES),
        hide_color=0,
        num_samples=500,  # <<< REDUCED SAMPLES
        random_seed=42 # Add for reproducibility
    )

    # Predict once outside LIME to get the top class reliably
    preds = model.predict(image[np.newaxis, ...])
    pred_class = np.argmax(preds[0])

    temp, mask = explanation.get_image_and_mask(
        pred_class,
        positive_only=True,
        num_features=5, # <<< REDUCED FEATURES
        hide_rest=False,
        min_weight=0.05 # Add a minimum weight threshold
    )

    temp_uint8 = (temp * 255).astype(np.uint8)
    # Ensure mask is boolean or 0/1
    mask_bool = mask > 0
    # Use mode='inner' or 'outer' for thinner/thicker boundaries if needed
    boundaries = mark_boundaries(temp_uint8, mask_bool, color=(0, 1, 0), mode='thick') # Green boundaries (BGR) 110 yellow

    return boundaries # Returns float 0-1 range image

# --- Processing Function (called by explain routes) ---
def generate_explanations(image_raw, model, pred_class, layer_name=None):
    """Generates GradCAM, IG, and LIME explanations for a single model."""
    results = {}

    # Grad-CAM
    if layer_name:
        try:
            grad = grad_cam(image_raw, model, pred_class, layer_name)
            # Apply brain mask *before* overlaying
            grad_masked = np.clip(grad * get_brain_mask(image_raw), 0, 1)
            grad_heatmap = apply_matplotlib_colormap(grad_masked)
            grad_overlay = cv2.addWeighted((image_raw * 255).astype(np.uint8), 0.6, grad_heatmap, 0.4, 0)
            results["gradcam"] = encode_image_to_base64(cv2.cvtColor(grad_overlay, cv2.COLOR_BGR2RGB))
        except Exception as e:
            print(f"Error generating Grad-CAM for {model.name}: {e}")
            results["gradcam"] = None # Or a placeholder image
    else:
        results["gradcam"] = None # No layer name provided

    # Integrated Gradients
    try:
        ig = integrated_gradients(image_raw, model, pred_class, steps=25) # Use tuned steps
        # Optional: Apply gamma correction if needed (can sometimes help visibility)
        # ig = ig ** 0.6
        ig_heatmap = apply_matplotlib_colormap(ig) # IG already includes brain mask
        ig_overlay = cv2.addWeighted((image_raw * 255).astype(np.uint8), 0.5, ig_heatmap, 0.5, 0) # Adjusted weights
        results["ig"] = encode_image_to_base64(cv2.cvtColor(ig_overlay, cv2.COLOR_BGR2RGB))
    except Exception as e:
        print(f"Error generating Integrated Gradients for {model.name}: {e}")
        results["ig"] = None

    # LIME
    try:
        # LIME returns image with boundaries, already float 0-1
        lime_img = lime_explainer(image_raw, model)
        # Convert to RGB before encoding
        lime_img_rgb = cv2.cvtColor((lime_img * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
        results["lime"] = encode_image_to_base64(lime_img_rgb)
    except Exception as e:
        print(f"Error generating LIME for {model.name}: {e}")
        results["lime"] = None

    return results


# --- Routes ---

@app.route('/')
def home():
    return "XAI MRI & PET API for Alzheimer's detection is live!"

# --- (predict_mri remains the same) ---
@app.route('/predict_mri', methods=['POST'])
def predict_mri():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        filename = secure_filename(file.filename)
        path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(path)

        img_input, _ = preprocess_image(path) # Only need input tensor here

        pred1 = mri_densenet.predict(img_input)[0]
        pred2 = mri_resnet.predict(img_input)[0]
        ensemble = (pred1 + pred2) / 2
        pred_idx = np.argmax(ensemble)
        pred_label = CLASSES[pred_idx]
        confidence = round(float(ensemble[pred_idx]) * 100, 2)
        all_confidences = {CLASSES[i]: round(float(ensemble[i] * 100), 2) for i in range(len(CLASSES))}

        # Store path associated with this prediction *temporarily*.
        # A better approach would be to return the filename and have the explain
        # endpoint accept the filename, or use a session/cache.

        app.config['LAST_MRI_PATH'] = path
        app.config['LAST_MRI_PRED_IDX'] = pred_idx # Store index too

        return jsonify({
            "predicted_label": pred_label,
            "confidence": confidence,
            "actual_label": "Unknown", # Assuming no ground truth provided
            "all_confidences": all_confidences
        })
    except ValueError as ve:
         return jsonify({"error": f"Image processing error: {str(ve)}"}), 400
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


# --- UPDATED /explain_mri with Parallelism ---
@app.route('/explain_mri', methods=['GET'])
def explain_mri():
    # Retrieve path and prediction from the prediction step
    # WARNING: Prone to race conditions with multiple users. See warning above.
    if 'LAST_MRI_PATH' not in app.config or 'LAST_MRI_PRED_IDX' not in app.config:
         return jsonify({"error": "Please run prediction first or file path not found."}), 400

    path = app.config['LAST_MRI_PATH']
    # Use the ensemble prediction index for consistency if desired,
    # or predict again for each model (current approach uses individual model preds)
    # ensemble_pred_idx = app.config['LAST_MRI_PRED_IDX'] # Example if using ensemble index

    try:
        _, img_raw = preprocess_image(path) # Need the raw-like image for explanations

        # Get individual predictions for explanation targeting
        img_input, _ = preprocess_image(path) # Need input tensor for prediction
        pred1_idx = np.argmax(mri_densenet.predict(img_input)[0])
        pred2_idx = np.argmax(mri_resnet.predict(img_input)[0])

        # Use ThreadPoolExecutor to run explanations in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit tasks: args=(image, model, predicted_class_index, layer_name)
            future1 = executor.submit(generate_explanations, img_raw, mri_densenet, pred1_idx, 'conv5_block32_concat')
            future2 = executor.submit(generate_explanations, img_raw, mri_resnet, pred2_idx, 'conv5_block3_out')

            # Get results
            densenet_output = future1.result()
            resnet_output = future2.result()

        # Structure the response
        response = {
            "gradcam": {"densenet": densenet_output.get("gradcam"), "resnet": resnet_output.get("gradcam")},
            "ig": {"densenet": densenet_output.get("ig"), "resnet": resnet_output.get("ig")},
            "lime": {"densenet": densenet_output.get("lime"), "resnet": resnet_output.get("lime")}
        }
        return jsonify(response)

    except FileNotFoundError:
         return jsonify({"error": f"Image file not found at {path}. Please predict again."}), 404
    except ValueError as ve:
         return jsonify({"error": f"Image processing error: {str(ve)}"}), 400
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": f"Explanation failed: {str(e)}"}), 500


# --- (predict_pet remains similar, add path storage) ---
@app.route('/predict_pet', methods=['POST'])
def predict_pet():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        filename = secure_filename(file.filename)
        path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(path)

        img_input, _ = preprocess_image(path)
        pred = pet_model.predict(img_input)[0]
        pred_idx = np.argmax(pred)
        pred_label = CLASSES[pred_idx]
        confidence = round(float(pred[pred_idx]) * 100, 2)
        all_confidences = {CLASSES[i]: round(float(pred[i] * 100), 2) for i in range(len(CLASSES))}

        # Store path and prediction index
        app.config['LAST_PET_PATH'] = path
        app.config['LAST_PET_PRED_IDX'] = pred_idx

        return jsonify({
            "predicted_label": pred_label,
            "confidence": confidence,
            "actual_label": "Unknown",
            "all_confidences": all_confidences
        })
    except ValueError as ve:
         return jsonify({"error": f"Image processing error: {str(ve)}"}), 400
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

# --- (explain_pet remains similar, uses generate_explanations) ---
@app.route('/explain_pet', methods=['GET'])
def explain_pet():
    if 'LAST_PET_PATH' not in app.config or 'LAST_PET_PRED_IDX' not in app.config:
         return jsonify({"error": "Please run PET prediction first or file path not found."}), 400

    path = app.config['LAST_PET_PATH']
    pred_class = app.config['LAST_PET_PRED_IDX']

    try:
        _, img_raw = preprocess_image(path)

        # Generate explanations for the single PET model
        pet_output = generate_explanations(img_raw, pet_model, pred_class, 'block5_conv4') # Ensure layer name is correct for VGG

        return jsonify(pet_output) # Return directly as it's just one model

    except FileNotFoundError:
         return jsonify({"error": f"Image file not found at {path}. Please predict again."}), 404
    except ValueError as ve:
         return jsonify({"error": f"Image processing error: {str(ve)}"}), 400
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": f"Explanation failed: {str(e)}"}), 500

# --- (predict_both remains similar, add path storage) ---
@app.route('/predict_both', methods=['POST'])
def predict_both():
    if 'mri' not in request.files or 'pet' not in request.files:
        return jsonify({"error": "Missing 'mri' or 'pet' file part"}), 400

    mri_file = request.files['mri']
    pet_file = request.files['pet']

    if mri_file.filename == '' or pet_file.filename == '':
        return jsonify({"error": "No selected file for MRI or PET"}), 400

    try:
        mri_filename = secure_filename("mri_" + mri_file.filename)
        pet_filename = secure_filename("pet_" + pet_file.filename)
        mri_path = os.path.join(UPLOAD_FOLDER, mri_filename)
        pet_path = os.path.join(UPLOAD_FOLDER, pet_filename)
        mri_file.save(mri_path)
        pet_file.save(pet_path)

        mri_input, _ = preprocess_image(mri_path)
        pet_input, _ = preprocess_image(pet_path)

        mri_pred1 = mri_densenet.predict(mri_input)[0]
        mri_pred2 = mri_resnet.predict(mri_input)[0]
        mri_ensemble = (mri_pred1 + mri_pred2) / 2
        pet_pred = pet_model.predict(pet_input)[0]

        final_probs = (mri_ensemble + pet_pred) / 2
        fused_pred_idx = np.argmax(final_probs)
        mri_pred_idx = np.argmax(mri_ensemble)
        pet_pred_idx = np.argmax(pet_pred)

        # Store paths and indices
        app.config['LAST_BOTH_MRI_PATH'] = mri_path
        app.config['LAST_BOTH_PET_PATH'] = pet_path
        app.config['LAST_BOTH_MRI_DENSE_IDX'] = np.argmax(mri_pred1) # Store individual for explanation
        app.config['LAST_BOTH_MRI_RES_IDX'] = np.argmax(mri_pred2) # Store individual for explanation
        app.config['LAST_BOTH_PET_IDX'] = pet_pred_idx

        return jsonify({
            "fused": {
                "predicted_label": CLASSES[fused_pred_idx],
                "confidence": round(float(final_probs[fused_pred_idx]) * 100, 2),
                "all_confidences": {CLASSES[i]: round(float(final_probs[i] * 100), 2) for i in range(len(CLASSES))}
            },
            "mri": {
                "predicted_label": CLASSES[mri_pred_idx],
                "confidence": round(float(np.max(mri_ensemble)) * 100, 2),
                "all_confidences": {CLASSES[i]: round(float(mri_ensemble[i] * 100), 2) for i in range(len(CLASSES))}
            },
            "pet": {
                "predicted_label": CLASSES[pet_pred_idx],
                "confidence": round(float(np.max(pet_pred)) * 100, 2),
                "all_confidences": {CLASSES[i]: round(float(pet_pred[i] * 100), 2) for i in range(len(CLASSES))}
            }
        })

    except ValueError as ve:
         return jsonify({"error": f"Image processing error: {str(ve)}"}), 400
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


# --- UPDATED /explain_both with Parallelism ---
@app.route('/explain_both', methods=['GET'])
def explain_both():
    if ('LAST_BOTH_MRI_PATH' not in app.config or
        'LAST_BOTH_PET_PATH' not in app.config or
        'LAST_BOTH_MRI_DENSE_IDX' not in app.config or
        'LAST_BOTH_MRI_RES_IDX' not in app.config or
        'LAST_BOTH_PET_IDX' not in app.config):
        return jsonify({"error": "Please run 'predict_both' first or required data not found."}), 400

    mri_path = app.config['LAST_BOTH_MRI_PATH']
    pet_path = app.config['LAST_BOTH_PET_PATH']
    mri_pred1_idx = app.config['LAST_BOTH_MRI_DENSE_IDX']
    mri_pred2_idx = app.config['LAST_BOTH_MRI_RES_IDX']
    pet_pred_idx = app.config['LAST_BOTH_PET_IDX']

    try:
        _, mri_raw = preprocess_image(mri_path)
        _, pet_raw = preprocess_image(pet_path)

        # Use ThreadPoolExecutor for all 3 explanations
        with ThreadPoolExecutor(max_workers=3) as executor:
             # Submit MRI tasks
            future_mri_dense = executor.submit(generate_explanations, mri_raw, mri_densenet, mri_pred1_idx, 'conv5_block32_concat')
            future_mri_resnet = executor.submit(generate_explanations, mri_raw, mri_resnet, mri_pred2_idx, 'conv5_block3_out')
            # Submit PET task
            future_pet = executor.submit(generate_explanations, pet_raw, pet_model, pet_pred_idx, 'block5_conv4')

            # Get results
            mri_dense_out = future_mri_dense.result()
            mri_resnet_out = future_mri_resnet.result()
            pet_out = future_pet.result()

        # Structure the response
        return jsonify({
            "mri": {
                "densenet": mri_dense_out,
                "resnet": mri_resnet_out
            },
            "pet": pet_out
        })

    except FileNotFoundError:
         return jsonify({"error": "MRI or PET image file not found. Please predict again."}), 404
    except ValueError as ve:
         return jsonify({"error": f"Image processing error: {str(ve)}"}), 400
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": f"Explanation failed: {str(e)}"}), 500


@app.route('/validate_mri', methods=['POST'])
def validate_mri():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    try:
        filename = secure_filename(file.filename)
        temp_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(temp_path)

        is_valid = validate_image_with_model(temp_path, mri_validator)

        os.remove(temp_path)  # Delete temp file after checking

        return jsonify({"is_valid": bool(is_valid)})  #  Wrap inside dict
    except Exception as e:
        print(f"Validation error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/validate_pet', methods=['POST'])
def validate_pet():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    try:
        filename = secure_filename(file.filename)
        temp_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(temp_path)

        is_valid = validate_image_with_model(temp_path, pet_validator)  # <<< Use PET Validator model here

        os.remove(temp_path)  # Delete temp file after checking

        return jsonify({"is_valid": bool(is_valid)})  # Always wrap in dict
    except Exception as e:
        print(f"Validation error: {str(e)}")
        return jsonify({"error": str(e)}), 500




if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True) # Set debug=False for production/performance test, threaded=True helps
