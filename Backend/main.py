# from flask import Flask, request, jsonify, send_file
# import os
# import numpy as np
# import tensorflow as tf
# import cv2
# from werkzeug.utils import secure_filename
# import matplotlib.pyplot as plt
# from lime import lime_image
# from skimage.segmentation import mark_boundaries
# # # #
# # # # # Initialize Flask app
# # # # app = Flask(__name__)
# # # #
# # # # # Load trained model
# # # # MODEL_PATH = "C:/Users/zunth/Downloads/MRI-3D-AXIAL_CBAM_DenseNet201.h5" # Ensure this is the correct model path
# # # # model = tf.keras.models.load_model(MODEL_PATH)
# # # # CLASSES = ['AD', 'CN', 'EMCI']
# # # # IMG_SIZE = 224
# # # # UPLOAD_FOLDER = "uploads"
# # # # EXPLANATION_FOLDER = "explanations"
# # # # os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# # # # os.makedirs(EXPLANATION_FOLDER, exist_ok=True)
# # # #
# # # # # Root route to confirm server is running
# # # # @app.route('/')
# # # # def home():
# # # #     return "Alzheimer's Detection API is running!"
# # # #
# # # # # Function to preprocess image
# # # # def preprocess_image(image_path):
# # # #     img = cv2.imread(image_path, cv2.IMREAD_COLOR)
# # # #     img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
# # # #     img = img.astype(np.float32) / 255.0  # Normalize
# # # #     return np.expand_dims(img, axis=0)
# # # #
# # # # # Endpoint to predict the uploaded image
# # # # @app.route('/predict', methods=['POST'])
# # # # def predict():
# # # #     if 'file' not in request.files:
# # # #         return jsonify({"error": "No file uploaded"}), 400
# # # #
# # # #     file = request.files['file']
# # # #     filename = secure_filename(file.filename)
# # # #     file_path = os.path.join(UPLOAD_FOLDER, filename)
# # # #     file.save(file_path)
# # # #
# # # #     # Preprocess and predict
# # # #     img = preprocess_image(file_path)
# # # #     predictions = model.predict(img)[0]
# # # #     predicted_index = np.argmax(predictions)
# # # #     predicted_label = CLASSES[predicted_index]
# # # #     confidence = float(predictions[predicted_index] * 100)
# # # #
# # # #     return jsonify({
# # # #         "actual_label": "Unknown",  # No ground truth available for uploaded image
# # # #         "predicted_label": predicted_label,
# # # #         "confidence": round(confidence, 2)
# # # #     })
# # # #
# # # # # Function to generate LIME explanation
# # # # def generate_lime(image_path):
# # # #     img = cv2.imread(image_path, cv2.IMREAD_COLOR)
# # # #     img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
# # # #     img = img.astype(np.float32) / 255.0
# # # #     explainer = lime_image.LimeImageExplainer()
# # # #     explanation = explainer.explain_instance(
# # # #         img, model.predict, top_labels=1, num_features=8, hide_rest=False
# # # #     )
# # # #     temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=8, hide_rest=False)
# # # #     lime_image_path = os.path.join(EXPLANATION_FOLDER, "lime.png")
# # # #     plt.imsave(lime_image_path, mark_boundaries(temp, mask))
# # # #     return lime_image_path
# # # #
# # # # # Endpoint for LIME explanation
# # # # @app.route('/explain/lime', methods=['GET'])
# # # # def explain_lime():
# # # #     last_uploaded = next(os.scandir(UPLOAD_FOLDER)).path  # Get the last uploaded file
# # # #     lime_path = generate_lime(last_uploaded)
# # # #     return send_file(lime_path, mimetype='image/png')
# # # #
# # # # # Function to generate Integrated Gradients (IG)
# # # # def generate_ig(image_path):
# # # #     img = preprocess_image(image_path)
# # # #     with tf.GradientTape() as tape:
# # # #         tape.watch(img)
# # # #         prediction = model(img)
# # # #         class_prediction = prediction[:, np.argmax(prediction)]
# # # #     gradients = tape.gradient(class_prediction, img)
# # # #     attribution_map = np.mean(np.abs(gradients.numpy()[0]), axis=-1)
# # # #     attribution_map -= attribution_map.min()
# # # #     attribution_map /= (attribution_map.max() + 1e-8)
# # # #     ig_image_path = os.path.join(EXPLANATION_FOLDER, "ig.png")
# # # #     plt.imsave(ig_image_path, attribution_map, cmap='jet')
# # # #     return ig_image_path
# # # #
# # # # # Endpoint for IG explanation
# # # # @app.route('/explain/ig', methods=['GET'])
# # # # def explain_ig():
# # # #     last_uploaded = next(os.scandir(UPLOAD_FOLDER)).path
# # # #     ig_path = generate_ig(last_uploaded)
# # # #     return send_file(ig_path, mimetype='image/png')
# # # #
# # # # if __name__ == '__main__':
# # # #     app.run(host='0.0.0.0', port=5000)  # Run locally
# # #
# # #
# from flask import Flask, request, jsonify, send_file
# from flask_cors import CORS
# import os
# import numpy as np
# import tensorflow as tf
# import cv2
# from werkzeug.utils import secure_filename
# import matplotlib.pyplot as plt
# from lime import lime_image
# from skimage.segmentation import mark_boundaries
# # # #
# # # # # Initialize Flask app
# # # # app = Flask(__name__)
# # # # CORS(app)  # Enable CORS for all routes
# # # #
# # # # # Load trained model
# # # # MODEL_PATH = "D:/SAI/MRI-3D-AXIAL_DenseNet201_CBAM_T1.h5" # Ensure this is the correct model path
# # # #
# # # # # "D:\SAI\MRI-3D-AXIAL_DenseNet201_CBAM_T1.h5"
# # # # model = tf.keras.models.load_model(MODEL_PATH)
# # # # CLASSES = ['AD', 'CN', 'EMCI']
# # # # IMG_SIZE = 224
# # # # UPLOAD_FOLDER = "uploads"
# # # # EXPLANATION_FOLDER = "explanations"
# # # # os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# # # # os.makedirs(EXPLANATION_FOLDER, exist_ok=True)
# # # #
# # # #
# # # # # Root route to confirm server is running
# # # # @app.route('/')
# # # # def home():
# # # #     return "Alzheimer's Detection API is running!"
# # # #
# # # #
# # # # # Function to preprocess image
# # # # def preprocess_image(image_path):
# # # #     img = cv2.imread(image_path, cv2.IMREAD_COLOR)
# # # #     img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
# # # #     img = img.astype(np.float32) / 255.0  # Normalize
# # # #     return np.expand_dims(img, axis=0)  # Ensure batch dimension
# # # #
# # # #
# # # # # Endpoint to predict the uploaded image
# # # # @app.route('/predict', methods=['POST'])
# # # # def predict():
# # # #     if 'file' not in request.files:
# # # #         return jsonify({"error": "No file uploaded"}), 400
# # # #
# # # #     file = request.files['file']
# # # #     filename = secure_filename(file.filename)
# # # #     file_path = os.path.join(UPLOAD_FOLDER, filename)
# # # #     file.save(file_path)
# # # #
# # # #     # Preprocess and predict
# # # #     img = preprocess_image(file_path)
# # # #     predictions = model.predict(img)[0]
# # # #     predicted_index = np.argmax(predictions)
# # # #     predicted_label = CLASSES[predicted_index]
# # # #     confidence = float(predictions[predicted_index] * 100)
# # # #
# # # #     return jsonify({
# # # #         "actual_label": "Unknown",  # No ground truth available for uploaded image
# # # #         "predicted_label": predicted_label,
# # # #         "confidence": round(confidence, 2)
# # # #     })
# # # #
# # # #
# # # # # Function to generate LIME explanation
# # # # def generate_lime(image_path):
# # # #     try:
# # # #         if not os.path.exists(image_path):
# # # #             return None  # File does not exist
# # # #
# # # #         img = cv2.imread(image_path, cv2.IMREAD_COLOR)
# # # #         img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
# # # #         img = img.astype(np.float32) / 255.0
# # # #
# # # #         explainer = lime_image.LimeImageExplainer()
# # # #         explanation = explainer.explain_instance(
# # # #             img, model.predict, top_labels=1, num_features=8
# # # #         )
# # # #         temp, mask = explanation.get_image_and_mask(
# # # #             explanation.top_labels[0], positive_only=True, num_features=8
# # # #         )
# # # #         lime_image_path = os.path.join(EXPLANATION_FOLDER, "lime.png")
# # # #         plt.imsave(lime_image_path, mark_boundaries(temp, mask))
# # # #         return lime_image_path
# # # #     except Exception as e:
# # # #         print(f"Error in LIME generation: {e}")
# # # #         return None
# # # #
# # # #
# # # # # Endpoint for LIME explanation
# # # # @app.route('/explain/lime', methods=['GET'])
# # # # def explain_lime():
# # # #     try:
# # # #         files = list(os.scandir(UPLOAD_FOLDER))
# # # #         if not files:
# # # #             return jsonify({"error": "No uploaded images found."}), 400
# # # #         last_uploaded = files[-1].path  # Get last uploaded file
# # # #         lime_path = generate_lime(last_uploaded)
# # # #         if lime_path:
# # # #             return send_file(lime_path, mimetype='image/png')
# # # #         else:
# # # #             return jsonify({"error": "Failed to generate LIME explanation."}), 500
# # # #     except Exception as e:
# # # #         return jsonify({"error": f"Server error: {e}"}), 500
# # # #
# # # #
# # # # # Function to generate Integrated Gradients (IG)
# # # # def generate_ig(image_path):
# # # #     try:
# # # #         img = preprocess_image(image_path)
# # # #         img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)  # Ensure Tensor format
# # # #
# # # #         with tf.GradientTape() as tape:
# # # #             tape.watch(img_tensor)
# # # #             prediction = model(img_tensor)
# # # #             class_prediction = prediction[:, np.argmax(prediction)]
# # # #         gradients = tape.gradient(class_prediction, img_tensor)
# # # #         attribution_map = np.mean(np.abs(gradients.numpy()[0]), axis=-1)
# # # #         attribution_map -= attribution_map.min()
# # # #         attribution_map /= (attribution_map.max() + 1e-8)
# # # #         ig_image_path = os.path.join(EXPLANATION_FOLDER, "ig.png")
# # # #         plt.imsave(ig_image_path, attribution_map, cmap='jet')
# # # #         return ig_image_path
# # # #     except Exception as e:
# # # #         print(f"Error in IG generation: {e}")
# # # #         return None
# # # #
# # # #
# # # # # Endpoint for IG explanation
# # # # @app.route('/explain/ig', methods=['GET'])
# # # # def explain_ig():
# # # #     try:
# # # #         files = list(os.scandir(UPLOAD_FOLDER))
# # # #         if not files:
# # # #             return jsonify({"error": "No uploaded images found."}), 400
# # # #         last_uploaded = files[-1].path  # Get last uploaded file
# # # #         ig_path = generate_ig(last_uploaded)
# # # #         if ig_path:
# # # #             return send_file(ig_path, mimetype='image/png')
# # # #         else:
# # # #             return jsonify({"error": "Failed to generate IG explanation."}), 500
# # # #     except Exception as e:
# # # #         return jsonify({"error": f"Server error: {e}"}), 500
# # # #
# # # #
# # # # if __name__ == '__main__':
# # # #     app.run(host='0.0.0.0', port=5000)  # Run locally
# # #
# from flask import Flask, request, jsonify, send_file
# from flask_cors import CORS
# import os
# import numpy as np
# import tensorflow as tf
# import cv2
# from werkzeug.utils import secure_filename
# import matplotlib.pyplot as plt
# from lime import lime_image
# from skimage.segmentation import mark_boundaries
# # # #
# # # # # Initialize Flask app
# # # # app = Flask(__name__)
# # # # CORS(app)  # Enable CORS for all routes
# # # #
# # # # # Load trained model
# # # # MODEL_PATH = "D:/SAI/MRI-3D-AXIAL_DenseNet201_CBAM_T1.h5" # Ensure this is the correct model path
# # # # model = tf.keras.models.load_model(MODEL_PATH)
# # # # CLASSES = ['AD', 'CN', 'EMCI', 'LMCI']
# # # # IMG_SIZE = 224
# # # # UPLOAD_FOLDER = "uploads"
# # # # EXPLANATION_FOLDER = "explanations"
# # # # os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# # # # os.makedirs(EXPLANATION_FOLDER, exist_ok=True)
# # # #
# # # # # Root route
# # # # @app.route('/')
# # # # def home():
# # # #     return "Alzheimer's Detection API is running!"
# # # #
# # # # # Function to preprocess image
# # # # def preprocess_image(image_path):
# # # #     img = cv2.imread(image_path, cv2.IMREAD_COLOR)
# # # #     img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
# # # #     img = img.astype(np.float32) / 255.0  # Normalize
# # # #     return np.expand_dims(img, axis=0)  # Add batch dimension
# # # #
# # # # # Prediction endpoint
# # # # @app.route('/predict', methods=['POST'])
# # # # def predict():
# # # #     if 'file' not in request.files:
# # # #         return jsonify({"error": "No file uploaded"}), 400
# # # #
# # # #     file = request.files['file']
# # # #     filename = secure_filename(file.filename)
# # # #     file_path = os.path.join(UPLOAD_FOLDER, filename)
# # # #     file.save(file_path)
# # # #
# # # #     # Preprocess and predict
# # # #     img = preprocess_image(file_path)
# # # #     predictions = model.predict(img)[0]
# # # #     predicted_index = np.argmax(predictions)
# # # #     predicted_label = CLASSES[predicted_index]
# # # #     confidence = float(predictions[predicted_index] * 100)
# # # #
# # # #     return jsonify({
# # # #         "actual_label": "Unknown",
# # # #         "predicted_label": predicted_label,
# # # #         "confidence": round(confidence, 2)
# # # #     })
# # # #
# # # # # Function to generate LIME explanation
# # # # def generate_lime(image_path):
# # # #     try:
# # # #         if not os.path.exists(image_path):
# # # #             return None
# # # #
# # # #         img = cv2.imread(image_path, cv2.IMREAD_COLOR)
# # # #         img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
# # # #         img = img.astype(np.float32) / 255.0
# # # #
# # # #         explainer = lime_image.LimeImageExplainer()
# # # #         explanation = explainer.explain_instance(
# # # #             img, model.predict, top_labels=1, num_features=8
# # # #         )
# # # #         temp, mask = explanation.get_image_and_mask(
# # # #             explanation.top_labels[0], positive_only=True, num_features=8
# # # #         )
# # # #         lime_image_path = os.path.join(EXPLANATION_FOLDER, "lime.png")
# # # #         plt.imsave(lime_image_path, mark_boundaries(temp, mask))
# # # #         return lime_image_path
# # # #     except Exception as e:
# # # #         print(f"Error in LIME generation: {e}")
# # # #         return None
# # # #
# # # # # LIME explanation endpoint
# # # # @app.route('/explain/lime', methods=['GET'])
# # # # def explain_lime():
# # # #     try:
# # # #         files = list(os.scandir(UPLOAD_FOLDER))
# # # #         if not files:
# # # #             return jsonify({"error": "No uploaded images found."}), 400
# # # #         last_uploaded = files[-1].path
# # # #         lime_path = generate_lime(last_uploaded)
# # # #         if lime_path:
# # # #             return send_file(lime_path, mimetype='image/png')
# # # #         else:
# # # #             return jsonify({"error": "Failed to generate LIME explanation."}), 500
# # # #     except Exception as e:
# # # #         return jsonify({"error": f"Server error: {e}"}), 500
# # # #
# # # # # **Updated Function: Generate Integrated Gradients (IG) with Enhanced Visibility**
# # # # def generate_ig(image_path):
# # # #     try:
# # # #         img = preprocess_image(image_path)
# # # #         img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
# # # #
# # # #         with tf.GradientTape() as tape:
# # # #             tape.watch(img_tensor)
# # # #             prediction = model(img_tensor)
# # # #             class_prediction = prediction[:, np.argmax(prediction)]
# # # #         gradients = tape.gradient(class_prediction, img_tensor)
# # # #
# # # #         # Compute absolute mean across RGB channels
# # # #         attribution_map = np.mean(np.abs(gradients.numpy()[0]), axis=-1)
# # # #
# # # #         # Normalize the attribution map
# # # #         attribution_map -= attribution_map.min()
# # # #         attribution_map /= (attribution_map.max() + 1e-8)
# # # #
# # # #         # Apply colormap for better visualization
# # # #         heatmap = cv2.applyColorMap(np.uint8(255 * attribution_map), cv2.COLORMAP_JET)
# # # #
# # # #         # Load original image for overlay
# # # #         original_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
# # # #         original_img = cv2.resize(original_img, (IMG_SIZE, IMG_SIZE))
# # # #
# # # #         # Blend heatmap with original image
# # # #         overlay = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)
# # # #
# # # #         # Save IG explanation
# # # #         ig_image_path = os.path.join(EXPLANATION_FOLDER, "ig_overlay.png")
# # # #         cv2.imwrite(ig_image_path, overlay)
# # # #
# # # #         return ig_image_path
# # # #     except Exception as e:
# # # #         print(f"Error in IG generation: {e}")
# # # #         return None
# # # #
# # # # # IG explanation endpoint
# # # # @app.route('/explain/ig', methods=['GET'])
# # # # def explain_ig():
# # # #     try:
# # # #         files = list(os.scandir(UPLOAD_FOLDER))
# # # #         if not files:
# # # #             return jsonify({"error": "No uploaded images found."}), 400
# # # #         last_uploaded = files[-1].path
# # # #         ig_path = generate_ig(last_uploaded)
# # # #         if ig_path:
# # # #             return send_file(ig_path, mimetype='image/png')
# # # #         else:
# # # #             return jsonify({"error": "Failed to generate IG explanation."}), 500
# # # #     except Exception as e:
# # # #         return jsonify({"error": f"Server error: {e}"}), 500
# # # #
# # # # if __name__ == '__main__':
# # # #     app.run(host='0.0.0.0', port=5000)  # Run locally
# # #
# # #
# # # # from flask import Flask, request, jsonify, send_file
# # # # from flask_cors import CORS
# # # # import os
# # # # import numpy as np
# # # # import tensorflow as tf
# # # # import cv2
# # # # from werkzeug.utils import secure_filename
# # # # import matplotlib.pyplot as plt
# # # # from lime import lime_image
# # # # from skimage.segmentation import mark_boundaries
# # # # import random
# # # #
# # # # # Initialize Flask app
# # # # app = Flask(__name__)
# # # # CORS(app)  # Enable CORS for all routes
# # # #
# # # # # Configuration
# # # # MODEL_PATH = "D:/SAI/MRI_DenseNet201_CBAM_best.h5" # Replace with your model path
# # # # CLASSES = ['AD', 'CN', 'EMCI', 'LMCI']
# # # # IMG_SIZE = 128
# # # # UPLOAD_FOLDER = "uploads"
# # # # EXPLANATION_FOLDER = "explanations"
# # # # os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# # # # os.makedirs(EXPLANATION_FOLDER, exist_ok=True)
# # # #
# # # # # Load trained model
# # # # model = tf.keras.models.load_model(MODEL_PATH)
# # # #
# # # #
# # # # # --- Helper Functions ---
# # # # def preprocess_image(image_path):
# # # #     img = cv2.imread(image_path, cv2.IMREAD_COLOR)
# # # #     img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
# # # #     img = img.astype(np.float32) / 255.0  # Normalize
# # # #     return np.expand_dims(img, axis=0)  # Add batch dimension
# # # #
# # # # def predict_class(image):
# # # #     predictions = model.predict(image)[0]
# # # #     predicted_index = np.argmax(predictions)
# # # #     predicted_label = CLASSES[predicted_index]
# # # #     confidence = float(predictions[predicted_index] * 100)
# # # #     return predicted_label, confidence
# # # #
# # # # # --- Explanation Functions ---
# # # # def lime_explainer(image, model):
# # # #     explainer = lime_image.LimeImageExplainer()
# # # #     explanation = explainer.explain_instance(
# # # #         image.astype('double'),
# # # #         model.predict,
# # # #         top_labels=len(CLASSES),
# # # #         hide_color=0,
# # # #         num_samples=1000
# # # #     )
# # # #     predicted_class = np.argmax(model.predict(image[np.newaxis, ...]))
# # # #     temp, mask = explanation.get_image_and_mask(
# # # #         predicted_class,
# # # #         positive_only=True,
# # # #         num_features=5,
# # # #         hide_rest=False
# # # #     )
# # # #     return mark_boundaries(temp, mask, color=(1, 1, 0))
# # # #
# # # # def grad_cam(image, model, class_idx, layer_name='conv5_block32_concat'):  # Adjust layer_name as needed
# # # #     image_tensor = tf.convert_to_tensor(image[np.newaxis, ...], dtype=tf.float32)
# # # #
# # # #     grad_model = tf.keras.models.Model(
# # # #         [model.inputs], [model.get_layer(layer_name).output, model.output]
# # # #     )
# # # #
# # # #     with tf.GradientTape() as tape:
# # # #         conv_outputs, predictions = grad_model(image_tensor)
# # # #         loss = predictions[:, class_idx]
# # # #
# # # #     grads = tape.gradient(loss, conv_outputs)
# # # #     pooled_grads = tf.reduce_mean(grads, axis=(1, 2))
# # # #
# # # #     heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
# # # #     heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
# # # #
# # # #     heatmap = tf.image.resize(heatmap[..., tf.newaxis], (IMG_SIZE, IMG_SIZE))
# # # #     heatmap = np.squeeze(heatmap)
# # # #
# # # #     return heatmap
# # # #
# # # # def integrated_gradients(image, model, class_idx):
# # # #     image_tensor = tf.convert_to_tensor(image[np.newaxis, ...], dtype=tf.float32)
# # # #
# # # #     with tf.GradientTape() as tape:
# # # #         tape.watch(image_tensor)
# # # #         preds = model(image_tensor)
# # # #         target = preds[:, class_idx]
# # # #
# # # #     grads = tape.gradient(target, image_tensor)
# # # #     attribution = np.mean(np.abs(grads.numpy().squeeze()), axis=-1)
# # # #
# # # #     # Normalize and mask brain region
# # # #     attribution = (attribution - attribution.min()) / (attribution.max() - attribution.min() + 1e-8)
# # # #     brain_mask = image.mean(axis=-1) > 0.1  # Adjust threshold based on your data
# # # #     return attribution * brain_mask
# # # #
# # # # # --- Explanation Generation and Saving ---
# # # # def generate_explanation_image(image, explanation_type):
# # # #     try:
# # # #         pred = model.predict(image[np.newaxis, ...])
# # # #         pred_class = np.argmax(pred)
# # # #
# # # #         if explanation_type == "lime":
# # # #             explanation = lime_explainer(image, model)
# # # #         elif explanation_type == "gradcam":
# # # #             heatmap = grad_cam(image, model, pred_class)
# # # #             explanation = image.copy()  # Make a copy of the original image
# # # #             heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
# # # #             heatmap = np.uint8(255 * heatmap)  # Scale to 0-255
# # # #             heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET) # Convert to heatmap
# # # #             explanation = cv2.addWeighted(explanation.astype(np.uint8), 0.6, heatmap.astype(np.uint8), 0.4, 0) # overlay
# # # #             explanation = cv2.cvtColor(explanation, cv2.COLOR_BGR2RGB) # convert to rgb
# # # #         elif explanation_type == "ig":
# # # #             heatmap = integrated_gradients(image, model, pred_class)
# # # #             explanation = image.copy()  # Make a copy of the original image
# # # #             heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
# # # #             heatmap = np.uint8(255 * heatmap)  # Scale to 0-255
# # # #             heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET) # Convert to heatmap
# # # #             explanation = cv2.addWeighted(explanation.astype(np.uint8), 0.6, heatmap.astype(np.uint8), 0.4, 0) # overlay
# # # #             explanation = cv2.cvtColor(explanation, cv2.COLOR_BGR2RGB) # convert to rgb
# # # #         else:
# # # #             return None, "Invalid explanation type"
# # # #
# # # #         explanation_path = os.path.join(EXPLANATION_FOLDER, f"{explanation_type}.png")
# # # #         plt.imsave(explanation_path, explanation)  # Save using Matplotlib
# # # #
# # # #         return explanation_path, None
# # # #
# # # #     except Exception as e:
# # # #         return None, str(e)
# # # #
# # # # # --- API Endpoints ---
# # # # @app.route('/')
# # # # def home():
# # # #     return "Alzheimer's Detection API is running!"
# # # #
# # # # @app.route('/predict', methods=['POST'])
# # # # def predict():
# # # #     if 'file' not in request.files:
# # # #         return jsonify({"error": "No file uploaded"}), 400
# # # #
# # # #     file = request.files['file']
# # # #     filename = secure_filename(file.filename)
# # # #     file_path = os.path.join(UPLOAD_FOLDER, filename)
# # # #     file.save(file_path)
# # # #
# # # #     img = preprocess_image(file_path)
# # # #     predicted_label, confidence = predict_class(img)
# # # #
# # # #     return jsonify({
# # # #         "actual_label": "Unknown",
# # # #         "predicted_label": predicted_label,
# # # #         "confidence": round(confidence, 2)
# # # #     })
# # # #
# # # # @app.route('/explain/<explanation_type>', methods=['GET'])
# # # # def explain(explanation_type):
# # # #     try:
# # # #         files = list(os.scandir(UPLOAD_FOLDER))
# # # #         if not files:
# # # #             return jsonify({"error": "No uploaded images found."}), 400
# # # #
# # # #         last_uploaded_path = files[-1].path # get the latest upload
# # # #         image = cv2.imread(last_uploaded_path, cv2.IMREAD_COLOR)
# # # #         image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
# # # #         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# # # #         explanation_path, error = generate_explanation_image(image, explanation_type)
# # # #         if explanation_path:
# # # #             return send_file(explanation_path, mimetype='image/png')
# # # #         else:
# # # #             return jsonify({"error": f"Failed to generate explanation: {error}"}), 500
# # # #
# # # #     except Exception as e:
# # # #         return jsonify({"error": f"Server error: {e}"}), 500
# # # #
# # # # # Remove or comment out this route because you no longer need test data
# # # # # # Add an endpoint to get a random image from test data for explanation
# # # # # @app.route('/explain/random/<explanation_type>', methods=['GET'])
# # # # # def explain_random(explanation_type):
# # # # #     try:
# # # # #         random_index = random.randint(0, len(test_images) - 1)
# # # # #         image = test_images[random_index]
# # # #
# # # # #         explanation_path, error = generate_explanation_image(image, explanation_type)
# # # #
# # # # #         if explanation_path:
# # # # #             return send_file(explanation_path, mimetype='image/png')
# # # # #         else:
# # # # #             return jsonify({"error": f"Failed to generate explanation: {error}"}), 500
# # # #
# # # # #     except Exception as e:
# # # # #         return jsonify({"error": f"Server error: {e}"}), 500
# # # #
# # # #
# # # # if __name__ == '__main__':
# # # #     app.run(host='0.0.0.0', port=5000, debug=True)
# # #
# # #
# # # # from flask import Flask, request, jsonify, send_file
# # # # from flask_cors import CORS
# # # # import os
# # # # import numpy as np
# # # # import tensorflow as tf
# # # # import cv2
# # # # from werkzeug.utils import secure_filename
# # # # import matplotlib.pyplot as plt
# # # # from lime import lime_image
# # # # from skimage.segmentation import mark_boundaries
# # # # import io  # Import the io module
# # # #
# # # # # Initialize Flask app
# # # # app = Flask(__name__)
# # # # CORS(app)  # Enable CORS for all routes
# # # #
# # # # # Configuration
# # # # MODEL_PATH = "D:/SAI/MRI_DenseNet201_CBAM_best.h5" # Replace with your model path
# # # # CLASSES = ['AD', 'CN', 'EMCI', 'LMCI']
# # # # IMG_SIZE = 128
# # # # UPLOAD_FOLDER = "uploads"
# # # # EXPLANATION_FOLDER = "explanations"
# # # # os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# # # # os.makedirs(EXPLANATION_FOLDER, exist_ok=True)
# # # #
# # # # # Load trained model
# # # # model = tf.keras.models.load_model(MODEL_PATH)
# # # #
# # # # # --- Helper Functions ---
# # # # def preprocess_image(image_path):
# # # #     img = cv2.imread(image_path, cv2.IMREAD_COLOR)
# # # #     img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
# # # #     img = img.astype(np.float32) / 255.0  # Normalize
# # # #     return np.expand_dims(img, axis=0)  # Add batch dimension
# # # #
# # # # def predict_class(image):
# # # #     predictions = model.predict(image)[0]
# # # #     predicted_index = np.argmax(predictions)
# # # #     predicted_label = CLASSES[predicted_index]
# # # #     confidence = float(predictions[predicted_index] * 100)
# # # #     return predicted_label, confidence
# # # #
# # # # # --- Explanation Functions ---
# # # # def lime_explainer(image, model):
# # # #     explainer = lime_image.LimeImageExplainer()
# # # #     explanation = explainer.explain_instance(
# # # #         image.astype('double'),
# # # #         model.predict,
# # # #         top_labels=len(CLASSES),
# # # #         hide_color=0,
# # # #         num_samples=1000
# # # #     )
# # # #     predicted_class = np.argmax(model.predict(image[np.newaxis, ...]))
# # # #     temp, mask = explanation.get_image_and_mask(
# # # #         predicted_class,
# # # #         positive_only=True,
# # # #         num_features=5,
# # # #         hide_rest=False
# # # #     )
# # # #     return mark_boundaries(temp, mask, color=(1, 1, 0))
# # # #
# # # # def grad_cam(image, model, class_idx, layer_name='conv5_block32_concat'):  # Adjust layer_name as needed
# # # #     image_tensor = tf.convert_to_tensor(image[np.newaxis, ...], dtype=tf.float32)
# # # #
# # # #     grad_model = tf.keras.models.Model(
# # # #         [model.inputs], [model.get_layer(layer_name).output, model.output]
# # # #     )
# # # #
# # # #     with tf.GradientTape() as tape:
# # # #         conv_outputs, predictions = grad_model(image_tensor)
# # # #         loss = predictions[:, class_idx]
# # # #
# # # #     grads = tape.gradient(loss, conv_outputs)
# # # #     pooled_grads = tf.reduce_mean(grads, axis=(1, 2))
# # # #
# # # #     heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
# # # #     heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
# # # #
# # # #     heatmap = tf.image.resize(heatmap[..., tf.newaxis], (IMG_SIZE, IMG_SIZE))
# # # #     heatmap = np.squeeze(heatmap)
# # # #
# # # #     return heatmap
# # # #
# # # # def integrated_gradients(image, model, class_idx):
# # # #     image_tensor = tf.convert_to_tensor(image[np.newaxis, ...], dtype=tf.float32)
# # # #
# # # #     with tf.GradientTape() as tape:
# # # #         tape.watch(image_tensor)
# # # #         preds = model(image_tensor)
# # # #         target = preds[:, class_idx]
# # # #
# # # #     grads = tape.gradient(target, image_tensor)
# # # #     attribution = np.mean(np.abs(grads.numpy().squeeze()), axis=-1)
# # # #
# # # #     # Normalize and mask brain region
# # # #     attribution = (attribution - attribution.min()) / (attribution.max() - attribution.min() + 1e-8)
# # # #     brain_mask = image.mean(axis=-1) > 0.1  # Adjust threshold based on your data
# # # #     return attribution * brain_mask
# # # #
# # # # # --- Explanation Generation and Encoding ---
# # # # def generate_explanation_images(image):
# # # #     try:
# # # #         pred = model.predict(image[np.newaxis, ...])
# # # #         pred_class = np.argmax(pred)
# # # #
# # # #         # Generate LIME
# # # #         lime_explanation = lime_explainer(image, model)
# # # #
# # # #         # Generate GradCAM
# # # #         heatmap_gradcam = grad_cam(image, model, pred_class)
# # # #         gradcam_explanation = image.copy()
# # # #         heatmap_gradcam = cv2.resize(heatmap_gradcam, (IMG_SIZE, IMG_SIZE))
# # # #         heatmap_gradcam = np.uint8(255 * heatmap_gradcam)
# # # #         heatmap_gradcam = cv2.applyColorMap(heatmap_gradcam, cv2.COLORMAP_JET)
# # # #         gradcam_explanation = cv2.addWeighted(gradcam_explanation.astype(np.uint8), 0.6, heatmap_gradcam.astype(np.uint8), 0.4, 0)
# # # #         gradcam_explanation = cv2.cvtColor(gradcam_explanation, cv2.COLOR_BGR2RGB)
# # # #
# # # #         # Generate Integrated Gradients
# # # #         heatmap_ig = integrated_gradients(image, model, pred_class)
# # # #         ig_explanation = image.copy()
# # # #         heatmap_ig = cv2.resize(heatmap_ig, (IMG_SIZE, IMG_SIZE))
# # # #         heatmap_ig = np.uint8(255 * heatmap_ig)
# # # #         heatmap_ig = cv2.applyColorMap(heatmap_ig, cv2.COLORMAP_JET)
# # # #         ig_explanation = cv2.addWeighted(ig_explanation.astype(np.uint8), 0.6, heatmap_ig.astype(np.uint8), 0.4, 0)
# # # #         ig_explanation = cv2.cvtColor(ig_explanation, cv2.COLOR_BGR2RGB)
# # # #
# # # #         # Encode the images as base64 strings
# # # #         lime_encoded = encode_image_to_base64(lime_explanation)
# # # #         gradcam_encoded = encode_image_to_base64(gradcam_explanation)
# # # #         ig_encoded = encode_image_to_base64(ig_explanation)
# # # #
# # # #         return {
# # # #             "lime": lime_encoded,
# # # #             "gradcam": gradcam_encoded,
# # # #             "ig": ig_encoded
# # # #         }, None
# # # #
# # # #     except Exception as e:
# # # #         return None, str(e)
# # # #
# # # # def encode_image_to_base64(image):
# # # #     """Encodes a numpy array (image) to a Base64 string."""
# # # #     try:
# # # #         # Convert image to uint8 if it's not already
# # # #         if image.dtype != np.uint8:
# # # #             image = (image * 255).astype(np.uint8)
# # # #
# # # #         # Use imencode to convert the image to a byte stream
# # # #         _, buffer = cv2.imencode('.png', image)
# # # #
# # # #         # Convert the buffer to an in-memory binary stream
# # # #         io_buf = io.BytesIO(buffer)
# # # #
# # # #         # Encode the binary stream to Base64
# # # #         encoded_string = base64.b64encode(io_buf.read()).decode('utf-8')
# # # #
# # # #         return encoded_string
# # # #     except Exception as e:
# # # #         print(f"Error encoding image to base64: {e}")
# # # #         return None
# # # #
# # # #
# # # # # --- API Endpoints ---
# # # # @app.route('/')
# # # # def home():
# # # #     return "Alzheimer's Detection API is running!"
# # # #
# # # # @app.route('/predict', methods=['POST'])
# # # # def predict():
# # # #     if 'file' not in request.files:
# # # #         return jsonify({"error": "No file uploaded"}), 400
# # # #
# # # #     file = request.files['file']
# # # #     filename = secure_filename(file.filename)
# # # #     file_path = os.path.join(UPLOAD_FOLDER, filename)
# # # #     file.save(file_path)
# # # #
# # # #     img = preprocess_image(file_path)
# # # #     predicted_label, confidence = predict_class(img)
# # # #
# # # #     return jsonify({
# # # #         "actual_label": "Unknown",
# # # #         "predicted_label": predicted_label,
# # # #         "confidence": round(confidence, 2)
# # # #     })
# # # #
# # # # import base64
# # # #
# # # # @app.route('/explain', methods=['GET'])
# # # # def explain():
# # # #     try:
# # # #         files = list(os.scandir(UPLOAD_FOLDER))
# # # #         if not files:
# # # #             return jsonify({"error": "No uploaded images found."}), 400
# # # #
# # # #         last_uploaded_path = files[-1].path  # get the latest upload
# # # #         image = cv2.imread(last_uploaded_path, cv2.IMREAD_COLOR)
# # # #         image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
# # # #         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# # # #
# # # #         explanations, error = generate_explanation_images(image)
# # # #         if explanations:
# # # #             return jsonify(explanations)  # Return the dictionary of encoded images
# # # #         else:
# # # #             return jsonify({"error": f"Failed to generate explanation: {error}"}), 500
# # # #
# # # #     except Exception as e:
# # # #         return jsonify({"error": f"Server error: {e}"}), 500
# # # #
# # # #
# # # # if __name__ == '__main__':
# # # #     app.run(host='0.0.0.0', port=5000, debug=True)
# # #
# # #
# # # # from flask import Flask, request, jsonify
# # # # from flask_cors import CORS
# # # # import os
# # # # import numpy as np
# # # # import tensorflow as tf
# # # # import cv2
# # # # from werkzeug.utils import secure_filename
# # # # import matplotlib.pyplot as plt
# # # # from lime import lime_image
# # # # from skimage.segmentation import mark_boundaries
# # # # import io
# # # # import base64
# # # #
# # # # # Initialize Flask app
# # # # app = Flask(__name__)
# # # # CORS(app)  # Enable CORS for all routes
# # # #
# # # # # Configuration
# # # # MODEL_PATH = "D:/SAI/MRI_DenseNet201_CBAM_best.h5" # Replace with your model path
# # # # CLASSES = ['AD', 'CN', 'EMCI', 'LMCI']
# # # # IMG_SIZE = 128
# # # # UPLOAD_FOLDER = "uploads"
# # # # EXPLANATION_FOLDER = "explanations"
# # # # os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# # # # os.makedirs(EXPLANATION_FOLDER, exist_ok=True)
# # # #
# # # # # Load trained model
# # # # model = tf.keras.models.load_model(MODEL_PATH)
# # # #
# # # # # --- Helper Functions ---
# # # # def preprocess_image(image_path):
# # # #     img = cv2.imread(image_path, cv2.IMREAD_COLOR)
# # # #     img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
# # # #     img = img.astype(np.float32) / 255.0  # Normalize
# # # #     return np.expand_dims(img, axis=0)  # Add batch dimension
# # # #
# # # # def predict_class(image):
# # # #     predictions = model.predict(image)[0]
# # # #     predicted_index = np.argmax(predictions)
# # # #     predicted_label = CLASSES[predicted_index]
# # # #     confidence = float(predictions[predicted_index] * 100)
# # # #     return predicted_label, confidence
# # # #
# # # # # --- Explanation Functions ---
# # # # def lime_explainer(image, model):
# # # #     explainer = lime_image.LimeImageExplainer()
# # # #     explanation = explainer.explain_instance(
# # # #         image.astype('double'),
# # # #         model.predict,
# # # #         top_labels=len(CLASSES),
# # # #         hide_color=0,
# # # #         num_samples=1000
# # # #     )
# # # #     predicted_class = np.argmax(model.predict(image[np.newaxis, ...]))
# # # #     temp, mask = explanation.get_image_and_mask(
# # # #         predicted_class,
# # # #         positive_only=True,
# # # #         num_features=5,
# # # #         hide_rest=False
# # # #     )
# # # #     return mark_boundaries(temp, mask, color=(1, 1, 0))
# # # #
# # # # def grad_cam(image, model, class_idx, layer_name='conv5_block32_concat'):
# # # #     image_tensor = tf.convert_to_tensor(image[np.newaxis, ...], dtype=tf.float32)
# # # #
# # # #     grad_model = tf.keras.models.Model(
# # # #         [model.inputs], [model.get_layer(layer_name).output, model.output]
# # # #     )
# # # #
# # # #     with tf.GradientTape() as tape:
# # # #         conv_outputs, predictions = grad_model(image_tensor)
# # # #         loss = predictions[:, class_idx]
# # # #
# # # #     grads = tape.gradient(loss, conv_outputs)
# # # #     pooled_grads = tf.reduce_mean(grads, axis=(1, 2))
# # # #
# # # #     heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
# # # #     heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
# # # #     heatmap = tf.image.resize(heatmap[..., tf.newaxis], (IMG_SIZE, IMG_SIZE))
# # # #     heatmap = np.squeeze(heatmap)
# # # #     return heatmap
# # # #
# # # # def integrated_gradients(image, model, class_idx):
# # # #     image_tensor = tf.convert_to_tensor(image[np.newaxis, ...], dtype=tf.float32)
# # # #
# # # #     with tf.GradientTape() as tape:
# # # #         tape.watch(image_tensor)
# # # #         preds = model(image_tensor)
# # # #         target = preds[:, class_idx]
# # # #
# # # #     grads = tape.gradient(target, image_tensor)
# # # #     attribution = np.mean(np.abs(grads.numpy().squeeze()), axis=-1)
# # # #
# # # #     # Normalize and mask brain region
# # # #     attribution = (attribution - attribution.min()) / (attribution.max() - attribution.min() + 1e-8)
# # # #     brain_mask = image.mean(axis=-1) > 0.1  # Adjust threshold based on your data
# # # #     return attribution * brain_mask
# # # #
# # # # def encode_image_to_base64(image):
# # # #     """Encodes a numpy array (image) to a Base64 string."""
# # # #     try:
# # # #         # Convert image to uint8 if it's not already
# # # #         if image.dtype != np.uint8:
# # # #             image = (image * 255).astype(np.uint8)
# # # #
# # # #         # Use imencode to convert the image to a byte stream
# # # #         _, buffer = cv2.imencode('.png', image)
# # # #
# # # #         # Convert the buffer to an in-memory binary stream
# # # #         io_buf = io.BytesIO(buffer)
# # # #
# # # #         # Encode the binary stream to Base64
# # # #         encoded_string = base64.b64encode(io_buf.read()).decode('utf-8')
# # # #
# # # #         return encoded_string
# # # #     except Exception as e:
# # # #         print(f"Error encoding image to base64: {e}")
# # # #         return None
# # # #
# # # # # Helper function to create brain mask
# # # # def get_brain_mask(image, threshold=0.1):
# # # #     """Creates a binary mask by thresholding low-intensity regions."""
# # # #     grayscale = np.mean(image, axis=-1) if image.ndim == 3 else image
# # # #     mask = grayscale > threshold
# # # #     return mask.astype(np.float32)
# # # #
# # # # # --- Explanation Generation and Encoding ---
# # # # def generate_explanation_images(image):
# # # #     try:
# # # #         pred = model.predict(image[np.newaxis, ...])
# # # #         pred_class = np.argmax(pred)
# # # #
# # # #         # Generate LIME
# # # #         lime_explanation = lime_explainer(image, model)
# # # #         lime_encoded = encode_image_to_base64(lime_explanation)
# # # #
# # # #         # Generate GradCAM with brain mask
# # # #         gradcam_heatmap = grad_cam(image, model, pred_class)
# # # #         brain_mask = get_brain_mask(image)
# # # #         masked_heatmap = gradcam_heatmap * brain_mask
# # # #         gradcam_explanation = image.copy()
# # # #         masked_heatmap = cv2.resize(masked_heatmap, (IMG_SIZE, IMG_SIZE))
# # # #         masked_heatmap = np.uint8(255 * masked_heatmap)
# # # #         masked_heatmap = cv2.applyColorMap(masked_heatmap, cv2.COLORMAP_JET)
# # # #         gradcam_explanation = cv2.addWeighted(gradcam_explanation.astype(np.uint8), 0.6, masked_heatmap.astype(np.uint8), 0.4, 0)
# # # #         gradcam_explanation = cv2.cvtColor(gradcam_explanation, cv2.COLOR_BGR2RGB)
# # # #         gradcam_encoded = encode_image_to_base64(gradcam_explanation)
# # # #
# # # #         # Generate Integrated Gradients
# # # #         ig_heatmap = integrated_gradients(image, model, pred_class)
# # # #         ig_explanation = image.copy()
# # # #         heatmap_ig = cv2.resize(ig_heatmap, (IMG_SIZE, IMG_SIZE))
# # # #         heatmap_ig = np.uint8(255 * heatmap_ig)
# # # #         heatmap_ig = cv2.applyColorMap(heatmap_ig, cv2.COLORMAP_JET)
# # # #         ig_explanation = cv2.addWeighted(ig_explanation.astype(np.uint8), 0.6, heatmap_ig.astype(np.uint8), 0.4, 0)
# # # #         ig_explanation = cv2.cvtColor(ig_explanation, cv2.COLOR_BGR2RGB)
# # # #         ig_encoded = encode_image_to_base64(ig_explanation)
# # # #
# # # #
# # # #         return {
# # # #             "lime": lime_encoded,
# # # #             "gradcam": gradcam_encoded,
# # # #             "ig": ig_encoded
# # # #         }, None
# # # #
# # # #     except Exception as e:
# # # #         return None, str(e)
# # # #
# # # # # --- API Endpoints ---
# # # # @app.route('/')
# # # # def home():
# # # #     return "Alzheimer's Detection API is running!"
# # # #
# # # # @app.route('/predict', methods=['POST'])
# # # # def predict():
# # # #     if 'file' not in request.files:
# # # #         return jsonify({"error": "No file uploaded"}), 400
# # # #
# # # #     file = request.files['file']
# # # #     filename = secure_filename(file.filename)
# # # #     file_path = os.path.join(UPLOAD_FOLDER, filename)
# # # #     file.save(file_path)
# # # #
# # # #     img = preprocess_image(file_path)
# # # #     predicted_label, confidence = predict_class(img)
# # # #
# # # #     return jsonify({
# # # #         "actual_label": "Unknown",
# # # #         "predicted_label": predicted_label,
# # # #         "confidence": round(confidence, 2)
# # # #     })
# # # #
# # # # @app.route('/explain', methods=['GET'])
# # # # def explain():
# # # #     try:
# # # #         files = list(os.scandir(UPLOAD_FOLDER))
# # # #         if not files:
# # # #             return jsonify({"error": "No uploaded images found."}), 400
# # # #
# # # #         last_uploaded_path = files[-1].path  # get the latest upload
# # # #         image = cv2.imread(last_uploaded_path, cv2.IMREAD_COLOR)
# # # #         image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
# # # #         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# # # #
# # # #         explanations, error = generate_explanation_images(image)
# # # #         if explanations:
# # # #             return jsonify(explanations)  # Return the dictionary of encoded images
# # # #         else:
# # # #             return jsonify({"error": f"Failed to generate explanation: {error}"}), 500
# # # #
# # # #     except Exception as e:
# # # #         return jsonify({"error": f"Server error: {e}"}), 500
# # # #
# # # #
# # # # if __name__ == '__main__':
# # # #     app.run(host='0.0.0.0', port=5000, debug=True)
# # #
# # #
# # # # from flask import Flask, request, jsonify
# # # # from flask_cors import CORS
# # # # import os
# # # # import numpy as np
# # # # import tensorflow as tf
# # # # import cv2
# # # # from werkzeug.utils import secure_filename
# # # # from lime import lime_image
# # # # from skimage.segmentation import mark_boundaries
# # # # import io
# # # # import base64
# # # # import logging
# # # #
# # # # # Initialize Flask app
# # # # app = Flask(__name__)
# # # # CORS(app)
# # # #
# # # # # Configure logging
# # # # logging.basicConfig(level=logging.DEBUG)
# # # #
# # # # # Configuration
# # # # MODEL_PATH = "D:/SAI/MRI_DenseNet201_CBAM_best.h5"
# # # # CLASSES = ['AD', 'CN', 'EMCI', 'LMCI']
# # # # IMG_SIZE = 128
# # # # UPLOAD_FOLDER = "uploads"
# # # # EXPLANATION_FOLDER = "explanations"
# # # # os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# # # # os.makedirs(EXPLANATION_FOLDER, exist_ok=True)
# # # #
# # # # # Load trained model
# # # # try:
# # # #     model = tf.keras.models.load_model(MODEL_PATH)
# # # #     app.logger.info("Model loaded successfully from %s", MODEL_PATH)
# # # # except Exception as e:
# # # #     app.logger.error("Error loading model: %s", e)
# # # #     raise
# # # #
# # # # # --- Helper Functions ---
# # # # def preprocess_image(image_path):
# # # #     try:
# # # #         img = cv2.imread(image_path, cv2.IMREAD_COLOR)
# # # #         if img is None:
# # # #             raise ValueError(f"Could not read image at {image_path}")
# # # #         img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
# # # #         img = img.astype(np.float32) / 255.0
# # # #         return np.expand_dims(img, axis=0)
# # # #     except Exception as e:
# # # #         app.logger.error(f"Error preprocessing image {image_path}: {e}")
# # # #         raise
# # # #
# # # # def predict_class(image):
# # # #     try:
# # # #         predictions = model.predict(image)[0]
# # # #         predicted_index = np.argmax(predictions)
# # # #         predicted_label = CLASSES[predicted_index]
# # # #         confidence = float(predictions[predicted_index] * 100)
# # # #         return predicted_label, confidence
# # # #     except Exception as e:
# # # #         app.logger.error(f"Error during prediction: {e}")
# # # #         raise
# # # #
# # # # # --- Explanation Functions ---
# # # # def lime_explainer(image, model):
# # # #     try:
# # # #         explainer = lime_image.LimeImageExplainer()
# # # #         explanation = explainer.explain_instance(
# # # #             image.astype('double'),
# # # #             model.predict,
# # # #             top_labels=len(CLASSES),
# # # #             hide_color=0,
# # # #             num_samples=1000
# # # #         )
# # # #         predicted_class = np.argmax(model.predict(image[np.newaxis, ...]))
# # # #         temp, mask = explanation.get_image_and_mask(
# # # #             predicted_class,
# # # #             positive_only=True,
# # # #             num_features=5,
# # # #             hide_rest=False
# # # #         )
# # # #         return mark_boundaries(temp, mask, color=(1, 1, 0))
# # # #     except Exception as e:
# # # #         app.logger.error(f"Error in LIME explanation: {e}")
# # # #         return np.zeros_like(image)
# # # #
# # # # def grad_cam(image, model, class_idx, layer_name='conv5_block32_concat'):
# # # #     try:
# # # #         image_tensor = tf.convert_to_tensor(image[np.newaxis, ...], dtype=tf.float32)
# # # #         grad_model = tf.keras.models.Model(
# # # #             [model.inputs], [model.get_layer(layer_name).output, model.output]
# # # #         )
# # # #         with tf.GradientTape() as tape:
# # # #             conv_outputs, predictions = grad_model(image_tensor)
# # # #             loss = predictions[:, class_idx]
# # # #         grads = tape.gradient(loss, conv_outputs)
# # # #         pooled_grads = tf.reduce_mean(grads, axis=(1, 2))
# # # #         heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
# # # #         heatmap = np.maximum(heatmap, 0)
# # # #         heatmap_max = np.max(heatmap)
# # # #         if heatmap_max == 0:
# # # #             app.logger.warning("Max heatmap value is 0. Skipping normalization.")
# # # #             return np.zeros((IMG_SIZE, IMG_SIZE))
# # # #         heatmap = heatmap / (heatmap_max + 1e-8)
# # # #         heatmap = tf.image.resize(heatmap[..., tf.newaxis], (IMG_SIZE, IMG_SIZE))
# # # #         heatmap = np.squeeze(heatmap)
# # # #         return heatmap
# # # #     except Exception as e:
# # # #         app.logger.error(f"Error in Grad-CAM explanation: {e}")
# # # #         return np.zeros((IMG_SIZE, IMG_SIZE))
# # # #
# # # # def integrated_gradients(image, model, class_idx):
# # # #     try:
# # # #         image_tensor = tf.convert_to_tensor(image[np.newaxis, ...], dtype=tf.float32)
# # # #         with tf.GradientTape() as tape:
# # # #             tape.watch(image_tensor)
# # # #             preds = model(image_tensor)
# # # #             target = preds[:, class_idx]
# # # #         grads = tape.gradient(target, image_tensor)
# # # #         attribution = np.mean(np.abs(grads.numpy().squeeze()), axis=-1)
# # # #         attribution = (attribution - attribution.min()) / (attribution.max() - attribution.min() + 1e-8)
# # # #         brain_mask = image.mean(axis=-1) > 0.1
# # # #         return attribution * brain_mask
# # # #     except Exception as e:
# # # #         app.logger.error(f"Error in IG explanation: {e}")
# # # #         return np.zeros((IMG_SIZE, IMG_SIZE))
# # # #
# # # # def encode_image_to_base64(image):
# # # #     try:
# # # #         if image.dtype != np.uint8:
# # # #             image = (image * 255).astype(np.uint8)
# # # #         _, buffer = cv2.imencode('.png', image)
# # # #         io_buf = io.BytesIO(buffer)
# # # #         encoded_string = base64.b64encode(io_buf.read()).decode('utf-8')
# # # #         return encoded_string
# # # #     except Exception as e:
# # # #         app.logger.error(f"Error encoding image to base64: {e}")
# # # #         return None
# # # #
# # # # def get_brain_mask(image, threshold=0.1):
# # # #     grayscale = np.mean(image, axis=-1) if image.ndim == 3 else image
# # # #     mask = grayscale > threshold
# # # #     return mask.astype(np.float32)
# # # #
# # # # # --- Explanation Generation and Encoding ---
# # # # def generate_explanation_images(image):
# # # #     try:
# # # #         pred = model.predict(image[np.newaxis, ...])
# # # #         pred_class = np.argmax(pred)
# # # #
# # # #         # LIME
# # # #         lime_explanation = lime_explainer(image, model)
# # # #         lime_encoded = encode_image_to_base64(lime_explanation)
# # # #
# # # #         # Grad-CAM
# # # #         gradcam_heatmap = grad_cam(image, model, pred_class)
# # # #         gradcam_heatmap_resized = cv2.resize(gradcam_heatmap, (IMG_SIZE, IMG_SIZE))
# # # #         brain_mask = get_brain_mask(image)
# # # #         masked_heatmap = gradcam_heatmap_resized * brain_mask
# # # #         masked_heatmap = np.nan_to_num(masked_heatmap)
# # # #         masked_heatmap = np.clip(masked_heatmap, 0, 1)
# # # #         gradcam_explanation = image.copy()
# # # #         heatmap_color = np.uint8(255 * masked_heatmap)
# # # #         heatmap_color = cv2.applyColorMap(heatmap_color, cv2.COLORMAP_JET)
# # # #         gradcam_explanation = cv2.addWeighted(gradcam_explanation.astype(np.uint8), 0.6, heatmap_color.astype(np.uint8), 0.4, 0)
# # # #         gradcam_explanation = cv2.cvtColor(gradcam_explanation, cv2.COLOR_BGR2RGB)
# # # #         gradcam_encoded = encode_image_to_base64(gradcam_explanation)
# # # #
# # # #         # Integrated Gradients
# # # #         ig_heatmap = integrated_gradients(image, model, pred_class)
# # # #         heatmap_ig = cv2.resize(ig_heatmap, (IMG_SIZE, IMG_SIZE))
# # # #         ig_colormap = np.uint8(255 * heatmap_ig)
# # # #         ig_colormap = cv2.applyColorMap(ig_colormap, cv2.COLORMAP_JET)
# # # #         ig_explanation = cv2.addWeighted(image.astype(np.uint8), 0.6, ig_colormap, 0.4, 0)
# # # #         ig_explanation = cv2.cvtColor(ig_explanation, cv2.COLOR_BGR2RGB)
# # # #         ig_encoded = encode_image_to_base64(ig_explanation)
# # # #
# # # #         return {
# # # #             "lime": lime_encoded,
# # # #             "gradcam": gradcam_encoded,
# # # #             "ig": ig_encoded
# # # #         }, None
# # # #     except Exception as e:
# # # #         app.logger.error(f"Error generating explanations: {e}")
# # # #         return None, str(e)
# # # #
# # # # # --- API Endpoints ---
# # # # @app.route('/')
# # # # def home():
# # # #     return "Alzheimer's Detection API is running!"
# # # #
# # # # @app.route('/predict', methods=['POST'])
# # # # def predict():
# # # #     try:
# # # #         if 'file' not in request.files:
# # # #             return jsonify({"error": "No file uploaded"}), 400
# # # #
# # # #         file = request.files['file']
# # # #         filename = secure_filename(file.filename)
# # # #         file_path = os.path.join(UPLOAD_FOLDER, filename)
# # # #         file.save(file_path)
# # # #
# # # #         img = preprocess_image(file_path)
# # # #         predicted_label, confidence = predict_class(img)
# # # #
# # # #         return jsonify({
# # # #             "actual_label": "Unknown",
# # # #             "predicted_label": predicted_label,
# # # #             "confidence": round(confidence, 2)
# # # #         })
# # # #     except Exception as e:
# # # #         app.logger.error(f"Error processing prediction request: {e}")
# # # #         return jsonify({"error": f"Server error: {e}"}), 500
# # # #
# # # # @app.route('/explain', methods=['GET'])
# # # # def explain():
# # # #     try:
# # # #         files = list(os.scandir(UPLOAD_FOLDER))
# # # #         if not files:
# # # #             return jsonify({"error": "No uploaded images found."}), 400
# # # #
# # # #         last_uploaded_path = files[-1].path
# # # #         app.logger.info(f"Using last uploaded file: {last_uploaded_path}")
# # # #
# # # #         image = cv2.imread(last_uploaded_path, cv2.IMREAD_COLOR)
# # # #         if image is None:
# # # #             return jsonify({"error": f"Failed to load the image file: {last_uploaded_path}"}), 500
# # # #
# # # #         image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
# # # #         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# # # #
# # # #         explanations, error = generate_explanation_images(image)
# # # #         if explanations:
# # # #             return jsonify(explanations)
# # # #         else:
# # # #             return jsonify({"error": f"Failed to generate explanation: {error}"}), 500
# # # #     except Exception as e:
# # # #         app.logger.error(f"Error processing explanation request: {e}")
# # # #         return jsonify({"error": f"Server error: {e}"}), 500
# # # #
# # # # if __name__ == '__main__':
# # # #     app.run(host='0.0.0.0', port=5000, debug=True)
# # #
# # # # from flask import Flask, request, jsonify
# # # # from flask_cors import CORS
# # # # import os
# # # # import numpy as np
# # # # import tensorflow as tf
# # # # import cv2
# # # # from werkzeug.utils import secure_filename
# # # # from lime import lime_image
# # # # from skimage.segmentation import mark_boundaries
# # # # import io
# # # # import base64
# # # # import logging
# # # #
# # # # # Initialize Flask app
# # # # app = Flask(__name__)
# # # # CORS(app)
# # # #
# # # # # Configure logging
# # # # logging.basicConfig(level=logging.DEBUG)
# # # #
# # # # # Configuration
# # # # MODEL_PATH = "D:/SAI/MRI_DenseNet201_CBAM_best.h5"
# # # # CLASSES = ['AD', 'CN', 'EMCI', 'LMCI']
# # # # IMG_SIZE = 128
# # # # UPLOAD_FOLDER = "uploads"
# # # # EXPLANATION_FOLDER = "explanations"
# # # # os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# # # # os.makedirs(EXPLANATION_FOLDER, exist_ok=True)
# # # #
# # # # # Load trained model
# # # # try:
# # # #     model = tf.keras.models.load_model(MODEL_PATH)
# # # #     app.logger.info("Model loaded successfully from %s", MODEL_PATH)
# # # # except Exception as e:
# # # #     app.logger.error("Error loading model: %s", e)
# # # #     raise
# # # #
# # # # # --- Helper Functions ---
# # # # def preprocess_image(image_path):
# # # #     try:
# # # #         img = cv2.imread(image_path, cv2.IMREAD_COLOR)
# # # #         if img is None:
# # # #             raise ValueError(f"Could not read image at {image_path}")
# # # #         img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
# # # #         img = img.astype(np.float32) / 255.0
# # # #         return np.expand_dims(img, axis=0)
# # # #     except Exception as e:
# # # #         app.logger.error(f"Error preprocessing image {image_path}: {e}")
# # # #         raise
# # # #
# # # # def predict_class(image):
# # # #     try:
# # # #         predictions = model.predict(image)[0]
# # # #         predicted_index = np.argmax(predictions)
# # # #         predicted_label = CLASSES[predicted_index]
# # # #         confidence = float(predictions[predicted_index] * 100)
# # # #         return predicted_label, confidence
# # # #     except Exception as e:
# # # #         app.logger.error(f"Error during prediction: {e}")
# # # #         raise
# # # #
# # # # # --- Explanation Functions ---
# # # # def lime_explainer(image, model):
# # # #     try:
# # # #         explainer = lime_image.LimeImageExplainer()
# # # #         explanation = explainer.explain_instance(
# # # #             image.astype('double'),
# # # #             model.predict,
# # # #             top_labels=len(CLASSES),
# # # #             hide_color=0,
# # # #             num_samples=1000
# # # #         )
# # # #         predicted_class = np.argmax(model.predict(image[np.newaxis, ...]))
# # # #         temp, mask = explanation.get_image_and_mask(
# # # #             predicted_class,
# # # #             positive_only=True,
# # # #             num_features=5,
# # # #             hide_rest=False
# # # #         )
# # # #         return mark_boundaries(temp, mask, color=(1, 1, 0))
# # # #     except Exception as e:
# # # #         app.logger.error(f"Error in LIME explanation: {e}")
# # # #         return np.zeros_like(image)
# # # #
# # # # def grad_cam(image, model, class_idx, layer_name='conv5_block32_concat'):
# # # #     try:
# # # #         image_tensor = tf.convert_to_tensor(image[np.newaxis, ...], dtype=tf.float32)
# # # #
# # # #         grad_model = tf.keras.models.Model(
# # # #             [model.inputs], [model.get_layer(layer_name).output, model.output]
# # # #         )
# # # #
# # # #         with tf.GradientTape() as tape:
# # # #             conv_outputs, predictions = grad_model(image_tensor)
# # # #             loss = predictions[:, class_idx]
# # # #
# # # #         grads = tape.gradient(loss, conv_outputs)
# # # #         pooled_grads = tf.reduce_mean(grads, axis=(1, 2))
# # # #
# # # #         heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
# # # #         heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
# # # #         heatmap = tf.image.resize(heatmap[..., tf.newaxis], (IMG_SIZE, IMG_SIZE))
# # # #         heatmap = np.squeeze(heatmap)
# # # #
# # # #         return heatmap
# # # #
# # # #     except Exception as e:
# # # #         app.logger.error(f"Error in Grad-CAM explanation: {e}")
# # # #         return np.zeros((IMG_SIZE, IMG_SIZE))
# # # #
# # # # def integrated_gradients(image, model, class_idx):
# # # #     try:
# # # #         image_tensor = tf.convert_to_tensor(image[np.newaxis, ...], dtype=tf.float32)
# # # #
# # # #         with tf.GradientTape() as tape:
# # # #             tape.watch(image_tensor)
# # # #             preds = model(image_tensor)
# # # #             target = preds[:, class_idx]
# # # #
# # # #         grads = tape.gradient(target, image_tensor)
# # # #         attribution = np.mean(np.abs(grads.numpy().squeeze()), axis=-1)
# # # #
# # # #         # Normalize and mask brain region
# # # #         attribution = (attribution - attribution.min()) / (attribution.max() - attribution.min() + 1e-8)
# # # #         brain_mask = image.mean(axis=-1) > 0.1  # Adjust threshold based on your data
# # # #         return attribution * brain_mask
# # # #     except Exception as e:
# # # #         app.logger.error(f"Error in IG explanation: {e}")
# # # #         return np.zeros((IMG_SIZE, IMG_SIZE))
# # # #
# # # # def encode_image_to_base64(image):
# # # #     try:
# # # #         if image.dtype != np.uint8:
# # # #             image = (image * 255).astype(np.uint8)
# # # #         _, buffer = cv2.imencode('.png', image)
# # # #         io_buf = io.BytesIO(buffer)
# # # #         encoded_string = base64.b64encode(io_buf.read()).decode('utf-8')
# # # #         return encoded_string
# # # #     except Exception as e:
# # # #         app.logger.error(f"Error encoding image to base64: {e}")
# # # #         return None
# # # #
# # # # def get_brain_mask(image, threshold=0.1):
# # # #     grayscale = np.mean(image, axis=-1) if image.ndim == 3 else image
# # # #     mask = grayscale > threshold
# # # #     return mask.astype(np.float32)
# # # #
# # # # # --- Explanation Generation and Encoding ---
# # # # def generate_explanation_images(image):
# # # #     try:
# # # #         pred = model.predict(image[np.newaxis, ...])
# # # #         pred_class = np.argmax(pred)
# # # #
# # # #         # LIME
# # # #         lime_explanation = lime_explainer(image, model)
# # # #         lime_encoded = encode_image_to_base64(lime_explanation)
# # # #
# # # #         # Grad-CAM
# # # #         gradcam_heatmap = grad_cam(image, model, pred_class)
# # # #         # Mask the Grad-CAM to only show inside the brain
# # # #         brain_mask = get_brain_mask(image)
# # # #         masked_heatmap = gradcam_heatmap * brain_mask
# # # #         gradcam_heatmap_resized = cv2.resize(masked_heatmap, (IMG_SIZE, IMG_SIZE)) #Resizing masked heat map
# # # #         gradcam_explanation = image.copy()
# # # #         heatmap_color = np.uint8(255 * gradcam_heatmap_resized)
# # # #         heatmap_color = cv2.applyColorMap(heatmap_color, cv2.COLORMAP_JET)
# # # #         gradcam_explanation = cv2.addWeighted(gradcam_explanation.astype(np.uint8), 0.6, heatmap_color.astype(np.uint8), 0.4, 0)
# # # #         gradcam_explanation = cv2.cvtColor(gradcam_explanation, cv2.COLOR_BGR2RGB)
# # # #         gradcam_encoded = encode_image_to_base64(gradcam_explanation)
# # # #
# # # #         # Integrated Gradients
# # # #         ig_heatmap = integrated_gradients(image, model, pred_class)
# # # #         ig_heatmap_resized = cv2.resize(ig_heatmap, (IMG_SIZE, IMG_SIZE))
# # # #         ig_colormap = np.uint8(255 * ig_heatmap_resized)
# # # #         ig_colormap = cv2.applyColorMap(ig_colormap, cv2.COLORMAP_JET)
# # # #         ig_explanation = cv2.addWeighted(image.astype(np.uint8), 0.6, ig_colormap, 0.4, 0)
# # # #         ig_explanation = cv2.cvtColor(ig_explanation, cv2.COLOR_BGR2RGB)
# # # #         ig_encoded = encode_image_to_base64(ig_explanation)
# # # #
# # # #         return {
# # # #             "lime": lime_encoded,
# # # #             "gradcam": gradcam_encoded,
# # # #             "ig": ig_encoded
# # # #         }, None
# # # #     except Exception as e:
# # # #         app.logger.error(f"Error generating explanations: {e}")
# # # #         return None, str(e)
# # # #
# # # # # --- API Endpoints ---
# # # # @app.route('/')
# # # # def home():
# # # #     return "Alzheimer's Detection API is running!"
# # # #
# # # # @app.route('/predict', methods=['POST'])
# # # # def predict():
# # # #     try:
# # # #         if 'file' not in request.files:
# # # #             return jsonify({"error": "No file uploaded"}), 400
# # # #
# # # #         file = request.files['file']
# # # #         filename = secure_filename(file.filename)
# # # #         file_path = os.path.join(UPLOAD_FOLDER, filename)
# # # #         file.save(file_path)
# # # #
# # # #         img = preprocess_image(file_path)
# # # #         predicted_label, confidence = predict_class(img)
# # # #
# # # #         return jsonify({
# # # #             "actual_label": "Unknown",
# # # #             "predicted_label": predicted_label,
# # # #             "confidence": round(confidence, 2)
# # # #         })
# # # #     except Exception as e:
# # # #         app.logger.error(f"Error processing prediction request: {e}")
# # # #         return jsonify({"error": f"Server error: {e}"}), 500
# # # #
# # # # @app.route('/explain', methods=['GET'])
# # # # def explain():
# # # #     try:
# # # #         files = list(os.scandir(UPLOAD_FOLDER))
# # # #         if not files:
# # # #             return jsonify({"error": "No uploaded images found."}), 400
# # # #
# # # #         last_uploaded_path = files[-1].path
# # # #         app.logger.info(f"Using last uploaded file: {last_uploaded_path}")
# # # #
# # # #         image = cv2.imread(last_uploaded_path, cv2.IMREAD_COLOR)
# # # #         if image is None:
# # # #             return jsonify({"error": f"Failed to load the image file: {last_uploaded_path}"}), 500
# # # #
# # # #         image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
# # # #         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# # # #
# # # #         explanations, error = generate_explanation_images(image)
# # # #         if explanations:
# # # #             return jsonify(explanations)
# # # #         else:
# # # #             return jsonify({"error": f"Failed to generate explanation: {error}"}), 500
# # # #     except Exception as e:
# # # #         app.logger.error(f"Error processing explanation request: {e}")
# # # #         return jsonify({"error": f"Server error: {e}"}), 500
# # # #
# # # # if __name__ == '__main__':
# # # #     app.run(host='0.0.0.0', port=5000, debug=True)
# # #
# # #
# # # # from flask import Flask, request, jsonify
# # # # from flask_cors import CORS
# # # # import os
# # # # import numpy as np
# # # # import tensorflow as tf
# # # # import cv2
# # # # from werkzeug.utils import secure_filename
# # # # from lime import lime_image
# # # # from skimage.segmentation import mark_boundaries
# # # # import io
# # # # import base64
# # # # import logging
# # # #
# # # # # Initialize Flask app
# # # # app = Flask(__name__)
# # # # CORS(app)
# # # #
# # # # # Configure logging
# # # # logging.basicConfig(level=logging.DEBUG)
# # # #
# # # # # Configuration
# # # # MODEL_PATH = "D:/SAI/MRI_DenseNet201_CBAM_best.h5"
# # # # CLASSES = ['AD', 'CN', 'EMCI', 'LMCI']
# # # # IMG_SIZE = 128
# # # # UPLOAD_FOLDER = "uploads"
# # # # EXPLANATION_FOLDER = "explanations"
# # # # os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# # # # os.makedirs(EXPLANATION_FOLDER, exist_ok=True)
# # # #
# # # # # Load trained model
# # # # try:
# # # #     model = tf.keras.models.load_model(MODEL_PATH)
# # # #     app.logger.info("Model loaded successfully from %s", MODEL_PATH)
# # # # except Exception as e:
# # # #     app.logger.error("Error loading model: %s", e)
# # # #     raise
# # # #
# # # # # --- Helper Functions ---
# # # # def preprocess_image(image_path):
# # # #     try:
# # # #         img = cv2.imread(image_path, cv2.IMREAD_COLOR)
# # # #         if img is None:
# # # #             raise ValueError(f"Could not read image at {image_path}")
# # # #         img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
# # # #         img = img.astype(np.float32) / 255.0
# # # #         return np.expand_dims(img, axis=0)
# # # #     except Exception as e:
# # # #         app.logger.error(f"Error preprocessing image {image_path}: {e}")
# # # #         raise
# # # #
# # # # def predict_class(image):
# # # #     try:
# # # #         predictions = model.predict(image)[0]
# # # #         predicted_index = np.argmax(predictions)
# # # #         predicted_label = CLASSES[predicted_index]
# # # #         confidence = float(predictions[predicted_index] * 100)
# # # #         return predicted_label, confidence
# # # #     except Exception as e:
# # # #         app.logger.error(f"Error during prediction: {e}")
# # # #         raise
# # # #
# # # # # --- Explanation Functions ---
# # # # def lime_explainer(image, model):
# # # #     try:
# # # #         explainer = lime_image.LimeImageExplainer()
# # # #         explanation = explainer.explain_instance(
# # # #             image.astype('double'),
# # # #             model.predict,
# # # #             top_labels=len(CLASSES),
# # # #             hide_color=0,
# # # #             num_samples=1000
# # # #         )
# # # #         predicted_class = np.argmax(model.predict(image[np.newaxis, ...]))
# # # #         temp, mask = explanation.get_image_and_mask(
# # # #             predicted_class,
# # # #             positive_only=True,
# # # #             num_features=5,
# # # #             hide_rest=False
# # # #         )
# # # #         return mark_boundaries(temp, mask, color=(1, 1, 0))
# # # #     except Exception as e:
# # # #         app.logger.error(f"Error in LIME explanation: {e}")
# # # #         return np.zeros_like(image)
# # # #
# # # # def grad_cam(image, model, class_idx, layer_name='conv5_block32_concat'):
# # # #     try:
# # # #         image_tensor = tf.convert_to_tensor(image[np.newaxis, ...], dtype=tf.float32)
# # # #
# # # #         grad_model = tf.keras.models.Model(
# # # #             [model.inputs], [model.get_layer(layer_name).output, model.output]
# # # #         )
# # # #
# # # #         with tf.GradientTape() as tape:
# # # #             conv_outputs, predictions = grad_model(image_tensor)
# # # #             loss = predictions[:, class_idx]
# # # #
# # # #         grads = tape.gradient(loss, conv_outputs)
# # # #         pooled_grads = tf.reduce_mean(grads, axis=(1, 2))
# # # #
# # # #         heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
# # # #         heatmap = np.maximum(heatmap, 0)
# # # #
# # # #         # Add this check to avoid division by zero
# # # #         if np.max(heatmap) == 0:
# # # #             app.logger.warning("Max heatmap value is 0 in Grad-CAM. Returning zero heatmap.")
# # # #             return np.zeros_like(heatmap)
# # # #
# # # #         heatmap = heatmap / np.max(heatmap)
# # # #         heatmap = tf.image.resize(heatmap[..., tf.newaxis], (IMG_SIZE, IMG_SIZE))
# # # #         heatmap = np.squeeze(heatmap)
# # # #
# # # #         return heatmap
# # # #
# # # #     except Exception as e:
# # # #         app.logger.error(f"Error in Grad-CAM explanation: {e}")
# # # #         return np.zeros((IMG_SIZE, IMG_SIZE))
# # # #
# # # # def integrated_gradients(image, model, class_idx):
# # # #     try:
# # # #         image_tensor = tf.convert_to_tensor(image[np.newaxis, ...], dtype=tf.float32)
# # # #
# # # #         with tf.GradientTape() as tape:
# # # #             tape.watch(image_tensor)
# # # #             preds = model(image_tensor)
# # # #             target = preds[:, class_idx]
# # # #
# # # #         grads = tape.gradient(target, image_tensor)
# # # #         attribution = np.mean(np.abs(grads.numpy().squeeze()), axis=-1)
# # # #
# # # #         # Normalize and mask brain region
# # # #         attribution = (attribution - attribution.min()) / (attribution.max() - attribution.min() + 1e-8)
# # # #         brain_mask = image.mean(axis=-1) > 0.1  # Adjust threshold based on your data
# # # #         return attribution * brain_mask
# # # #     except Exception as e:
# # # #         app.logger.error(f"Error in IG explanation: {e}")
# # # #         return np.zeros((IMG_SIZE, IMG_SIZE))
# # # #
# # # # def encode_image_to_base64(image):
# # # #     try:
# # # #         if image.dtype != np.uint8:
# # # #             image = (image * 255).astype(np.uint8)
# # # #         _, buffer = cv2.imencode('.png', image)
# # # #         io_buf = io.BytesIO(buffer)
# # # #         encoded_string = base64.b64encode(io_buf.read()).decode('utf-8')
# # # #         return encoded_string
# # # #     except Exception as e:
# # # #         app.logger.error(f"Error encoding image to base64: {e}")
# # # #         return None
# # # #
# # # # def get_brain_mask(image, threshold=0.1):
# # # #     grayscale = np.mean(image, axis=-1) if image.ndim == 3 else image
# # # #     mask = grayscale > threshold
# # # #     return mask.astype(np.float32)
# # # #
# # # # # --- Explanation Generation and Encoding ---
# # # # def generate_explanation_images(image):
# # # #     try:
# # # #         pred = model.predict(image[np.newaxis, ...])
# # # #         pred_class = np.argmax(pred)
# # # #
# # # #         # LIME
# # # #         lime_explanation = lime_explainer(image, model)
# # # #         lime_encoded = encode_image_to_base64(lime_explanation)
# # # #
# # # #         # Grad-CAM
# # # #         gradcam_heatmap = grad_cam(image, model, pred_class)
# # # #         # Mask the Grad-CAM to only show inside the brain
# # # #         brain_mask = get_brain_mask(image)
# # # #         masked_heatmap = gradcam_heatmap * brain_mask
# # # #         masked_heatmap = np.nan_to_num(masked_heatmap)  # Handle NaN values after masking
# # # #         masked_heatmap = np.clip(masked_heatmap, 0, 1)
# # # #         gradcam_heatmap_resized = cv2.resize(masked_heatmap, (IMG_SIZE, IMG_SIZE)) #Resizing masked heat map
# # # #
# # # #         gradcam_explanation = image.copy()
# # # #         heatmap_color = np.uint8(255 * gradcam_heatmap_resized)
# # # #         heatmap_color = cv2.applyColorMap(heatmap_color, cv2.COLORMAP_JET)
# # # #         gradcam_explanation = cv2.addWeighted(gradcam_explanation.astype(np.uint8), 0.6, heatmap_color.astype(np.uint8), 0.4, 0)
# # # #         gradcam_explanation = cv2.cvtColor(gradcam_explanation, cv2.COLOR_BGR2RGB)
# # # #         gradcam_encoded = encode_image_to_base64(gradcam_explanation)
# # # #
# # # #         # Integrated Gradients
# # # #         ig_heatmap = integrated_gradients(image, model, pred_class)
# # # #         ig_heatmap = np.nan_to_num(ig_heatmap) # Handle NaN in IG too
# # # #         ig_heatmap_resized = cv2.resize(ig_heatmap, (IMG_SIZE, IMG_SIZE))
# # # #         ig_colormap = np.uint8(255 * ig_heatmap_resized)
# # # #         ig_colormap = cv2.applyColorMap(ig_colormap, cv2.COLORMAP_JET)
# # # #         ig_explanation = cv2.addWeighted(image.astype(np.uint8), 0.6, ig_colormap, 0.4, 0)
# # # #         ig_explanation = cv2.cvtColor(ig_explanation, cv2.COLOR_BGR2RGB)
# # # #         ig_encoded = encode_image_to_base64(ig_explanation)
# # # #
# # # #         return {
# # # #             "lime": lime_encoded,
# # # #             "gradcam": gradcam_encoded,
# # # #             "ig": ig_encoded
# # # #         }, None
# # # #     except Exception as e:
# # # #         app.logger.error(f"Error generating explanations: {e}")
# # # #         return None, str(e)
# # # #
# # # # # --- API Endpoints ---
# # # # @app.route('/')
# # # # def home():
# # # #     return "Alzheimer's Detection API is running!"
# # # #
# # # # @app.route('/predict', methods=['POST'])
# # # # def predict():
# # # #     try:
# # # #         if 'file' not in request.files:
# # # #             return jsonify({"error": "No file uploaded"}), 400
# # # #
# # # #         file = request.files['file']
# # # #         filename = secure_filename(file.filename)
# # # #         file_path = os.path.join(UPLOAD_FOLDER, filename)
# # # #         file.save(file_path)
# # # #
# # # #         img = preprocess_image(file_path)
# # # #         predicted_label, confidence = predict_class(img)
# # # #
# # # #         return jsonify({
# # # #             "actual_label": "Unknown",
# # # #             "predicted_label": predicted_label,
# # # #             "confidence": round(confidence, 2)
# # # #         })
# # # #     except Exception as e:
# # # #         app.logger.error(f"Error processing prediction request: {e}")
# # # #         return jsonify({"error": f"Server error: {e}"}), 500
# # # #
# # # # @app.route('/explain', methods=['GET'])
# # # # def explain():
# # # #     try:
# # # #         files = list(os.scandir(UPLOAD_FOLDER))
# # # #         if not files:
# # # #             return jsonify({"error": "No uploaded images found."}), 400
# # # #
# # # #         last_uploaded_path = files[-1].path
# # # #         app.logger.info(f"Using last uploaded file: {last_uploaded_path}")
# # # #
# # # #         image = cv2.imread(last_uploaded_path, cv2.IMREAD_COLOR)
# # # #         if image is None:
# # # #             return jsonify({"error": f"Failed to load the image file: {last_uploaded_path}"}), 500
# # # #
# # # #         image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
# # # #         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# # # #
# # # #         explanations, error = generate_explanation_images(image)
# # # #         if explanations:
# # # #             return jsonify(explanations)
# # # #         else:
# # # #             return jsonify({"error": f"Failed to generate explanation: {error}"}), 500
# # # #     except Exception as e:
# # # #         app.logger.error(f"Error processing explanation request: {e}")
# # # #         return jsonify({"error": f"Server error: {e}"}), 500
# # # #
# # # # if __name__ == '__main__':
# # # #     app.run(host='0.0.0.0', port=5000, debug=True)
# # #
# # #
# # # # from flask import Flask, request, jsonify
# # # # from flask_cors import CORS
# # # # import os
# # # # import numpy as np
# # # # import tensorflow as tf
# # # # import cv2
# # # # from werkzeug.utils import secure_filename
# # # # from lime import lime_image
# # # # from skimage.segmentation import mark_boundaries
# # # # import io
# # # # import base64
# # # # import logging
# # # #
# # # # # Initialize Flask app
# # # # app = Flask(__name__)
# # # # CORS(app)
# # # #
# # # # # Configure logging
# # # # logging.basicConfig(level=logging.INFO)
# # # #
# # # # # Configuration
# # # # MODEL_PATH = "D:/SAI/MRI_DenseNet201_CBAM_best.h5"
# # # # CLASSES = ['AD', 'CN', 'EMCI', 'LMCI']
# # # # IMG_SIZE = 128
# # # # UPLOAD_FOLDER = "uploads"
# # # # os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# # # #
# # # # # Load trained model
# # # # model = tf.keras.models.load_model(MODEL_PATH)
# # # # app.logger.info("Model loaded from %s", MODEL_PATH)
# # # #
# # # # # --- Helper Functions ---
# # # # def preprocess_image(image_path):
# # # #     img = cv2.imread(image_path, cv2.IMREAD_COLOR)
# # # #     if img is None:
# # # #         raise ValueError(f"Could not read image: {image_path}")
# # # #     img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
# # # #     img = img.astype(np.float32) / 255.0
# # # #     return np.expand_dims(img, axis=0), img  # return both batched and unbatched
# # # #
# # # # def encode_image_to_base64(image):
# # # #     if image.dtype != np.uint8:
# # # #         image = (image * 255).astype(np.uint8)
# # # #     _, buffer = cv2.imencode('.png', image)
# # # #     io_buf = io.BytesIO(buffer)
# # # #     return base64.b64encode(io_buf.read()).decode('utf-8')
# # # #
# # # # def get_brain_mask(image, threshold=0.1):
# # # #     grayscale = np.mean(image, axis=-1)
# # # #     return (grayscale > threshold).astype(np.float32)
# # # #
# # # # # --- XAI Explanation Methods ---
# # # # def integrated_gradients(image, model, class_idx):
# # # #     image_tensor = tf.convert_to_tensor(image[np.newaxis, ...], dtype=tf.float32)
# # # #     with tf.GradientTape() as tape:
# # # #         tape.watch(image_tensor)
# # # #         preds = model(image_tensor)
# # # #         target = preds[:, class_idx]
# # # #     grads = tape.gradient(target, image_tensor)
# # # #     attribution = np.mean(np.abs(grads.numpy().squeeze()), axis=-1)
# # # #     attribution = (attribution - attribution.min()) / (attribution.max() - attribution.min() + 1e-8)
# # # #     brain_mask = get_brain_mask(image)
# # # #     return attribution * brain_mask
# # # #
# # # # def grad_cam(image, model, class_idx, layer_name='conv5_block32_concat'):
# # # #     image_tensor = tf.convert_to_tensor(image[np.newaxis, ...], dtype=tf.float32)
# # # #     grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])
# # # #     with tf.GradientTape() as tape:
# # # #         conv_outputs, predictions = grad_model(image_tensor)
# # # #         loss = predictions[:, class_idx]
# # # #     grads = tape.gradient(loss, conv_outputs)
# # # #     pooled_grads = tf.reduce_mean(grads, axis=(1, 2))
# # # #     heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
# # # #     heatmap = np.maximum(heatmap, 0)
# # # #     if np.max(heatmap) != 0:
# # # #         heatmap /= np.max(heatmap)
# # # #     heatmap = tf.image.resize(heatmap[..., tf.newaxis], (IMG_SIZE, IMG_SIZE))
# # # #     return np.squeeze(heatmap.numpy())
# # # #
# # # # def lime_explainer(image, model):
# # # #     explainer = lime_image.LimeImageExplainer()
# # # #     explanation = explainer.explain_instance(
# # # #         image.astype('double'),
# # # #         model.predict,
# # # #         top_labels=len(CLASSES),
# # # #         hide_color=0,
# # # #         num_samples=1000
# # # #     )
# # # #     pred_class = np.argmax(model.predict(image[np.newaxis, ...]))
# # # #     temp, mask = explanation.get_image_and_mask(
# # # #         pred_class,
# # # #         positive_only=True,
# # # #         num_features=5,
# # # #         hide_rest=False
# # # #     )
# # # #     return mark_boundaries(temp, mask, color=(1, 1, 0))
# # # #
# # # # # --- API Routes ---
# # # # @app.route('/')
# # # # def home():
# # # #     return "Alzheimer's XAI API is running!"
# # # #
# # # # @app.route('/predict', methods=['POST'])
# # # # def predict():
# # # #     try:
# # # #         if 'file' not in request.files:
# # # #             return jsonify({"error": "No file uploaded"}), 400
# # # #
# # # #         file = request.files['file']
# # # #         filename = secure_filename(file.filename)
# # # #         file_path = os.path.join(UPLOAD_FOLDER, filename)
# # # #         file.save(file_path)
# # # #
# # # #         img_input, img_raw = preprocess_image(file_path)
# # # #         predictions = model.predict(img_input)[0]
# # # #         pred_index = np.argmax(predictions)
# # # #         pred_label = CLASSES[pred_index]
# # # #         confidence = float(predictions[pred_index] * 100)
# # # #
# # # #         # Save last uploaded image path
# # # #         app.config['LAST_IMAGE_PATH'] = file_path
# # # #
# # # #         return jsonify({
# # # #             "actual_label": "Unknown",
# # # #             "predicted_label": pred_label,
# # # #             "confidence": round(confidence, 2)
# # # #         })
# # # #
# # # #     except Exception as e:
# # # #         app.logger.error(f"Prediction error: {e}")
# # # #         return jsonify({"error": str(e)}), 500
# # # #
# # # # @app.route('/explain', methods=['GET'])
# # # # def explain():
# # # #     try:
# # # #         if 'LAST_IMAGE_PATH' not in app.config:
# # # #             return jsonify({"error": "No uploaded image found."}), 400
# # # #
# # # #         file_path = app.config['LAST_IMAGE_PATH']
# # # #         img_input, img_raw = preprocess_image(file_path)
# # # #         img_raw_rgb = cv2.cvtColor((img_raw * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
# # # #         pred = model.predict(img_input)[0]
# # # #         pred_class = np.argmax(pred)
# # # #
# # # #         # LIME
# # # #         lime_img = lime_explainer(img_raw, model)
# # # #         lime_encoded = encode_image_to_base64(lime_img)
# # # #
# # # #         # Grad-CAM
# # # #         gradcam_heatmap = grad_cam(img_raw, model, pred_class)
# # # #         brain_mask = get_brain_mask(img_raw)
# # # #         masked_heatmap = gradcam_heatmap * brain_mask
# # # #         heatmap_img = cv2.applyColorMap(np.uint8(255 * masked_heatmap), cv2.COLORMAP_JET)
# # # #         gradcam_explanation = cv2.addWeighted((img_raw * 255).astype(np.uint8), 0.6, heatmap_img, 0.4, 0)
# # # #         gradcam_encoded = encode_image_to_base64(cv2.cvtColor(gradcam_explanation, cv2.COLOR_BGR2RGB))
# # # #
# # # #         # Integrated Gradients
# # # #         ig_heatmap = integrated_gradients(img_raw, model, pred_class)
# # # #         ig_colormap = cv2.applyColorMap(np.uint8(255 * ig_heatmap), cv2.COLORMAP_JET)
# # # #         ig_explanation = cv2.addWeighted((img_raw * 255).astype(np.uint8), 0.6, ig_colormap, 0.4, 0)
# # # #         ig_encoded = encode_image_to_base64(cv2.cvtColor(ig_explanation, cv2.COLOR_BGR2RGB))
# # # #
# # # #         return jsonify({
# # # #             "lime": lime_encoded,
# # # #             "gradcam": gradcam_encoded,
# # # #             "ig": ig_encoded
# # # #         })
# # # #
# # # #     except Exception as e:
# # # #         app.logger.error(f"Explanation error: {e}")
# # # #         return jsonify({"error": str(e)}), 500
# # # #
# # # # # Run the app
# # # # if __name__ == '__main__':
# # # #     app.run(host='0.0.0.0', port=5000, debug=True)
# # #
# # #
# # # # from flask import Flask, request, jsonify
# # # # from flask_cors import CORS
# # # # import os
# # # # import numpy as np
# # # # import tensorflow as tf
# # # # import cv2
# # # # from werkzeug.utils import secure_filename
# # # # from lime import lime_image
# # # # from skimage.segmentation import mark_boundaries
# # # # import io
# # # # import base64
# # # # import logging
# # # # import matplotlib.cm as cm
# # # #
# # # # # Initialize Flask app
# # # # app = Flask(__name__)
# # # # CORS(app)
# # # #
# # # # # Configuration
# # # # MODEL_PATH = "D:/SAI/MRI_DenseNet201_CBAM_best.h5"
# # # # CLASSES = ['AD', 'CN', 'EMCI', 'LMCI']
# # # # IMG_SIZE = 128
# # # # UPLOAD_FOLDER = "uploads"
# # # # os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# # # #
# # # # # Load model
# # # # model = tf.keras.models.load_model(MODEL_PATH)
# # # #
# # # # # Helper to encode image as base64
# # # # def encode_image_to_base64(image):
# # # #     if image.dtype != np.uint8:
# # # #         image = (image * 255).astype(np.uint8)
# # # #     _, buffer = cv2.imencode('.png', image)
# # # #     return base64.b64encode(buffer).decode('utf-8')
# # # #
# # # # # Helper to apply matplotlib colormap (jet)
# # # # def apply_matplotlib_colormap(heatmap):
# # # #     cmap = cm.get_cmap('jet')
# # # #     colored_heatmap = cmap(heatmap)[:, :, :3]  # drop alpha
# # # #     return (colored_heatmap * 255).astype(np.uint8)
# # # #
# # # # # Preprocessing
# # # # def preprocess_image(image_path):
# # # #     img = cv2.imread(image_path, cv2.IMREAD_COLOR)
# # # #     if img is None:
# # # #         raise ValueError("Cannot load image")
# # # #     img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
# # # #     img = img.astype(np.float32) / 255.0
# # # #     return np.expand_dims(img, axis=0), img  # batched and unbatched
# # # #
# # # # # Brain mask
# # # # def get_brain_mask(image, threshold=0.1):
# # # #     return (np.mean(image, axis=-1) > threshold).astype(np.float32)
# # # #
# # # # # XAI methods
# # # # def integrated_gradients(image, model, class_idx):
# # # #     image_tensor = tf.convert_to_tensor(image[np.newaxis, ...], dtype=tf.float32)
# # # #     with tf.GradientTape() as tape:
# # # #         tape.watch(image_tensor)
# # # #         preds = model(image_tensor)
# # # #         target = preds[:, class_idx]
# # # #     grads = tape.gradient(target, image_tensor)
# # # #     attribution = np.mean(np.abs(grads.numpy().squeeze()), axis=-1)
# # # #     attribution = (attribution - attribution.min()) / (attribution.max() - attribution.min() + 1e-8)
# # # #     brain_mask = get_brain_mask(image)
# # # #     return attribution * brain_mask
# # # #
# # # # def grad_cam(image, model, class_idx, layer_name='conv5_block32_concat'):
# # # #     image_tensor = tf.convert_to_tensor(image[np.newaxis, ...], dtype=tf.float32)
# # # #     grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])
# # # #     with tf.GradientTape() as tape:
# # # #         conv_outputs, predictions = grad_model(image_tensor)
# # # #         loss = predictions[:, class_idx]
# # # #     grads = tape.gradient(loss, conv_outputs)
# # # #     pooled_grads = tf.reduce_mean(grads, axis=(1, 2))
# # # #     heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1).numpy().squeeze()
# # # #     heatmap = np.maximum(heatmap, 0)
# # # #     if np.max(heatmap) != 0:
# # # #         heatmap = heatmap / np.max(heatmap)
# # # #     heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
# # # #     return heatmap
# # # #
# # # # def lime_explainer(image, model):
# # # #     explainer = lime_image.LimeImageExplainer()
# # # #     explanation = explainer.explain_instance(
# # # #         image.astype('double'),
# # # #         model.predict,
# # # #         top_labels=len(CLASSES),
# # # #         hide_color=0,
# # # #         num_samples=1000
# # # #     )
# # # #     pred_class = np.argmax(model.predict(image[np.newaxis, ...]))
# # # #     temp, mask = explanation.get_image_and_mask(pred_class, positive_only=True, num_features=5, hide_rest=False)
# # # #     return mark_boundaries(temp, mask, color=(1, 1, 0))
# # # #
# # # # # Routes
# # # # @app.route('/')
# # # # def home():
# # # #     return "XAI API for Alzheimer's detection is live!"
# # # #
# # # # @app.route('/predict', methods=['POST'])
# # # # def predict():
# # # #     try:
# # # #         if 'file' not in request.files:
# # # #             return jsonify({"error": "No file uploaded"}), 400
# # # #
# # # #         file = request.files['file']
# # # #         filename = secure_filename(file.filename)
# # # #         file_path = os.path.join(UPLOAD_FOLDER, filename)
# # # #         file.save(file_path)
# # # #
# # # #         img_input, img_raw = preprocess_image(file_path)
# # # #         predictions = model.predict(img_input)[0]
# # # #         pred_index = np.argmax(predictions)
# # # #         pred_label = CLASSES[pred_index]
# # # #         confidence = float(predictions[pred_index] * 100)
# # # #
# # # #         app.config['LAST_IMAGE_PATH'] = file_path
# # # #
# # # #         return jsonify({
# # # #             "actual_label": "Unknown",
# # # #             "predicted_label": pred_label,
# # # #             "confidence": round(confidence, 2)
# # # #         })
# # # #
# # # #     except Exception as e:
# # # #         return jsonify({"error": str(e)}), 500
# # # #
# # # # @app.route('/explain', methods=['GET'])
# # # # def explain():
# # # #     try:
# # # #         if 'LAST_IMAGE_PATH' not in app.config:
# # # #             return jsonify({"error": "No uploaded image yet"}), 400
# # # #
# # # #         file_path = app.config['LAST_IMAGE_PATH']
# # # #         img_input, img_raw = preprocess_image(file_path)
# # # #         pred = model.predict(img_input)[0]
# # # #         pred_class = np.argmax(pred)
# # # #
# # # #         # Grad-CAM
# # # #         gradcam_heatmap = grad_cam(img_raw, model, pred_class)
# # # #         brain_mask = get_brain_mask(img_raw)
# # # #         masked_gradcam = gradcam_heatmap * brain_mask
# # # #         gradcam_colored = apply_matplotlib_colormap(masked_gradcam)
# # # #         gradcam_overlay = cv2.addWeighted((img_raw * 255).astype(np.uint8), 0.6, gradcam_colored, 0.4, 0)
# # # #         gradcam_encoded = encode_image_to_base64(cv2.cvtColor(gradcam_overlay, cv2.COLOR_BGR2RGB))
# # # #
# # # #         # Integrated Gradients
# # # #         ig_heatmap = integrated_gradients(img_raw, model, pred_class)
# # # #         ig_colored = apply_matplotlib_colormap(ig_heatmap)
# # # #         ig_overlay = cv2.addWeighted((img_raw * 255).astype(np.uint8), 0.6, ig_colored, 0.4, 0)
# # # #         ig_encoded = encode_image_to_base64(cv2.cvtColor(ig_overlay, cv2.COLOR_BGR2RGB))
# # # #
# # # #         # LIME
# # # #         lime_img = lime_explainer(img_raw, model)
# # # #         lime_encoded = encode_image_to_base64((lime_img * 255).astype(np.uint8))
# # # #
# # # #         return jsonify({
# # # #             "lime": lime_encoded,
# # # #             "gradcam": gradcam_encoded,
# # # #             "ig": ig_encoded
# # # #         })
# # # #
# # # #     except Exception as e:
# # # #         return jsonify({"error": str(e)}), 500
# # # #
# # # # # Run app
# # # # if __name__ == '__main__':
# # # #     app.run(host='0.0.0.0', port=5000, debug=True)
# # # #
# # #
# # # # from flask import Flask, request, jsonify
# # # # from flask_cors import CORS
# # # # import os
# # # # import numpy as np
# # # # import tensorflow as tf
# # # # import cv2
# # # # from werkzeug.utils import secure_filename
# # # # from lime import lime_image
# # # # from skimage.segmentation import mark_boundaries
# # # # import io
# # # # import base64
# # # # import logging
# # # # import matplotlib.cm as cm
# # # #
# # # # # Initialize Flask app
# # # # app = Flask(__name__)
# # # # CORS(app)
# # # #
# # # # # Configuration
# # # # MODEL_PATH = "D:/SAI/MRI_DenseNet201_CBAM_best.h5"
# # # # CLASSES = ['AD', 'CN', 'EMCI', 'LMCI']
# # # # IMG_SIZE = 128
# # # # UPLOAD_FOLDER = "uploads"
# # # # os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# # # #
# # # # # Load model
# # # # model = tf.keras.models.load_model(MODEL_PATH)
# # # #
# # # # # Helper to encode image as base64
# # # # def encode_image_to_base64(image):
# # # #     try:
# # # #         if image.dtype != np.uint8:
# # # #             image = (image * 255).astype(np.uint8)
# # # #         _, buffer = cv2.imencode('.png', image)
# # # #         return base64.b64encode(buffer).decode('utf-8')
# # # #     except Exception as e:
# # # #         print("Base64 encoding failed:", e)
# # # #         return None
# # # #
# # # # # Helper to apply matplotlib colormap (jet)
# # # # def apply_matplotlib_colormap(heatmap):
# # # #     cmap = cm.get_cmap('jet')
# # # #     colored_heatmap = cmap(heatmap)[:, :, :3]  # drop alpha
# # # #     return (colored_heatmap * 255).astype(np.uint8)
# # # #
# # # # # Preprocessing
# # # # def preprocess_image(image_path):
# # # #     img = cv2.imread(image_path, cv2.IMREAD_COLOR)
# # # #     if img is None:
# # # #         raise ValueError("Cannot load image")
# # # #     img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
# # # #     img = img.astype(np.float32) / 255.0
# # # #     return np.expand_dims(img, axis=0), img  # batched and unbatched
# # # #
# # # # # Brain mask
# # # # def get_brain_mask(image, threshold=0.1):
# # # #     return (np.mean(image, axis=-1) > threshold).astype(np.float32)
# # # #
# # # # # XAI methods
# # # # def integrated_gradients(image, model, class_idx):
# # # #     try:
# # # #         image_tensor = tf.convert_to_tensor(image[np.newaxis, ...], dtype=tf.float32)
# # # #         with tf.GradientTape() as tape:
# # # #             tape.watch(image_tensor)
# # # #             preds = model(image_tensor)
# # # #             target = preds[:, class_idx]
# # # #         grads = tape.gradient(target, image_tensor)
# # # #         attribution = np.mean(np.abs(grads.numpy().squeeze()), axis=-1)
# # # #
# # # #         # Fix: Remove NaNs and clip
# # # #         attribution = np.nan_to_num(attribution)
# # # #         attribution = np.clip(attribution, 0, 1)
# # # #         if np.max(attribution) != 0:
# # # #             attribution = (attribution - np.min(attribution)) / (np.max(attribution) - np.min(attribution) + 1e-8)
# # # #         brain_mask = get_brain_mask(image)
# # # #         return attribution * brain_mask
# # # #     except Exception as e:
# # # #         print("Error generating IG:", e)
# # # #         return np.zeros((IMG_SIZE, IMG_SIZE))
# # # #
# # # # def grad_cam(image, model, class_idx, layer_name='conv5_block32_concat'):
# # # #     try:
# # # #         image_tensor = tf.convert_to_tensor(image[np.newaxis, ...], dtype=tf.float32)
# # # #         grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])
# # # #         with tf.GradientTape() as tape:
# # # #             conv_outputs, predictions = grad_model(image_tensor)
# # # #             loss = predictions[:, class_idx]
# # # #         grads = tape.gradient(loss, conv_outputs)
# # # #         pooled_grads = tf.reduce_mean(grads, axis=(1, 2))
# # # #         heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1).numpy().squeeze()
# # # #         heatmap = np.maximum(heatmap, 0)
# # # #         if np.max(heatmap) != 0:
# # # #             heatmap /= np.max(heatmap)
# # # #         heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
# # # #         return heatmap
# # # #     except Exception as e:
# # # #         print("Error generating Grad-CAM:", e)
# # # #         return np.zeros((IMG_SIZE, IMG_SIZE))
# # # #
# # # # def lime_explainer(image, model):
# # # #     try:
# # # #         explainer = lime_image.LimeImageExplainer()
# # # #         explanation = explainer.explain_instance(
# # # #             image.astype('double'),
# # # #             model.predict,
# # # #             top_labels=len(CLASSES),
# # # #             hide_color=0,
# # # #             num_samples=1000
# # # #         )
# # # #         pred_class = np.argmax(model.predict(image[np.newaxis, ...]))
# # # #         temp, mask = explanation.get_image_and_mask(pred_class, positive_only=True, num_features=5, hide_rest=False)
# # # #         return mark_boundaries(temp, mask, color=(1, 1, 0))
# # # #     except Exception as e:
# # # #         print("Error generating LIME:", e)
# # # #         return np.zeros_like(image)
# # # #
# # # # # Routes
# # # # @app.route('/')
# # # # def home():
# # # #     return "XAI API for Alzheimer's detection is live!"
# # # #
# # # # @app.route('/predict', methods=['POST'])
# # # # def predict():
# # # #     try:
# # # #         if 'file' not in request.files:
# # # #             return jsonify({"error": "No file uploaded"}), 400
# # # #
# # # #         file = request.files['file']
# # # #         filename = secure_filename(file.filename)
# # # #         file_path = os.path.join(UPLOAD_FOLDER, filename)
# # # #         file.save(file_path)
# # # #
# # # #         img_input, img_raw = preprocess_image(file_path)
# # # #         predictions = model.predict(img_input)[0]
# # # #         pred_index = np.argmax(predictions)
# # # #         pred_label = CLASSES[pred_index]
# # # #         confidence = float(predictions[pred_index] * 100)
# # # #
# # # #         app.config['LAST_IMAGE_PATH'] = file_path
# # # #
# # # #         return jsonify({
# # # #             "actual_label": "Unknown",
# # # #             "predicted_label": pred_label,
# # # #             "confidence": round(confidence, 2)
# # # #         })
# # # #
# # # #     except Exception as e:
# # # #         return jsonify({"error": str(e)}), 500
# # # #
# # # # @app.route('/explain', methods=['GET'])
# # # # def explain():
# # # #     try:
# # # #         if 'LAST_IMAGE_PATH' not in app.config:
# # # #             return jsonify({"error": "No uploaded image yet"}), 400
# # # #
# # # #         file_path = app.config['LAST_IMAGE_PATH']
# # # #         img_input, img_raw = preprocess_image(file_path)
# # # #         pred = model.predict(img_input)[0]
# # # #         pred_class = np.argmax(pred)
# # # #
# # # #         # Grad-CAM
# # # #         gradcam_heatmap = grad_cam(img_raw, model, pred_class)
# # # #         brain_mask = get_brain_mask(img_raw)
# # # #         masked_gradcam = gradcam_heatmap * brain_mask
# # # #         masked_gradcam = np.nan_to_num(masked_gradcam)
# # # #         masked_gradcam = np.clip(masked_gradcam, 0, 1)
# # # #         gradcam_colored = apply_matplotlib_colormap(masked_gradcam)
# # # #         gradcam_overlay = cv2.addWeighted((img_raw * 255).astype(np.uint8), 0.6, gradcam_colored, 0.4, 0)
# # # #         gradcam_encoded = encode_image_to_base64(cv2.cvtColor(gradcam_overlay, cv2.COLOR_BGR2RGB))
# # # #
# # # #         # Integrated Gradients
# # # #         ig_heatmap = integrated_gradients(img_raw, model, pred_class)
# # # #         print("IG Heatmap Stats - Max:", np.max(ig_heatmap), "Min:", np.min(ig_heatmap))
# # # #         ig_colored = apply_matplotlib_colormap(ig_heatmap)
# # # #         ig_overlay = cv2.addWeighted((img_raw * 255).astype(np.uint8), 0.6, ig_colored, 0.4, 0)
# # # #         ig_encoded = encode_image_to_base64(cv2.cvtColor(ig_overlay, cv2.COLOR_BGR2RGB))
# # # #         print("IG Encoded is None?", ig_encoded is None)
# # # #
# # # #         # LIME
# # # #         lime_img = lime_explainer(img_raw, model)
# # # #         lime_encoded = encode_image_to_base64((lime_img * 255).astype(np.uint8))
# # # #
# # # #         return jsonify({
# # # #             "lime": lime_encoded,
# # # #             "gradcam": gradcam_encoded,
# # # #             "ig": ig_encoded
# # # #         })
# # # #
# # # #     except Exception as e:
# # # #         return jsonify({"error": str(e)}), 500
# # # #
# # # # # Run app
# # # # if __name__ == '__main__':
# # # #     app.run(host='0.0.0.0', port=5000, debug=True)
# # # #
# # #
# # # #
# # # # from flask import Flask, request, jsonify
# # # # from flask_cors import CORS
# # # # import os
# # # # import numpy as np
# # # # import tensorflow as tf
# # # # import cv2
# # # # from werkzeug.utils import secure_filename
# # # # from lime import lime_image
# # # # from skimage.segmentation import mark_boundaries
# # # # import io
# # # # import base64
# # # # import logging
# # # # import matplotlib.cm as cm
# # # #
# # # # # Initialize Flask app
# # # # app = Flask(__name__)
# # # # CORS(app)
# # # #
# # # # # Configuration
# # # # MODEL_PATH = "D:/SAI/MRI_DenseNet201_CBAM_best.h5"
# # # # CLASSES = ['AD', 'CN', 'EMCI', 'LMCI']
# # # # IMG_SIZE = 128
# # # # UPLOAD_FOLDER = "uploads"
# # # # os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# # # #
# # # # # Load model
# # # # model = tf.keras.models.load_model(MODEL_PATH)
# # # #
# # # # # Helper to encode image as base64
# # # # def encode_image_to_base64(image):
# # # #     try:
# # # #         if image.dtype != np.uint8:
# # # #             image = (image * 255).astype(np.uint8)
# # # #         _, buffer = cv2.imencode('.png', image)
# # # #         return base64.b64encode(buffer).decode('utf-8')
# # # #     except Exception as e:
# # # #         print("Base64 encoding failed:", e)
# # # #         return None
# # # #
# # # # # Helper to apply matplotlib colormap (jet)
# # # # def apply_matplotlib_colormap(heatmap):
# # # #     cmap = cm.get_cmap('jet')
# # # #     colored_heatmap = cmap(heatmap)[:, :, :3]  # drop alpha
# # # #     return (colored_heatmap * 255).astype(np.uint8)
# # # #
# # # # # Preprocessing
# # # # def preprocess_image(image_path):
# # # #     img = cv2.imread(image_path, cv2.IMREAD_COLOR)
# # # #     if img is None:
# # # #         raise ValueError("Cannot load image")
# # # #     img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
# # # #     img = img.astype(np.float32) / 255.0
# # # #     return np.expand_dims(img, axis=0), img  # batched and unbatched
# # # #
# # # # # Brain mask
# # # # def get_brain_mask(image, threshold=0.1):
# # # #     return (np.mean(image, axis=-1) > threshold).astype(np.float32)
# # # #
# # # # # Integrated Gradients
# # # # def integrated_gradients(image, model, class_idx):
# # # #     try:
# # # #         image_tensor = tf.convert_to_tensor(image[np.newaxis, ...], dtype=tf.float32)
# # # #         with tf.GradientTape() as tape:
# # # #             tape.watch(image_tensor)
# # # #             preds = model(image_tensor)
# # # #             target = preds[:, class_idx]
# # # #         grads = tape.gradient(target, image_tensor)
# # # #         attribution = np.mean(np.abs(grads.numpy().squeeze()), axis=-1)
# # # #
# # # #         attribution = np.nan_to_num(attribution)
# # # #         attribution = np.clip(attribution, 0, 1)
# # # #         if np.max(attribution) != 0:
# # # #             attribution = (attribution - np.min(attribution)) / (np.max(attribution) - np.min(attribution) + 1e-8)
# # # #
# # # #         brain_mask = get_brain_mask(image)
# # # #         return attribution * brain_mask
# # # #     except Exception as e:
# # # #         print("Error generating IG:", e)
# # # #         return np.zeros((IMG_SIZE, IMG_SIZE))
# # # #
# # # # # Grad-CAM
# # # # def grad_cam(image, model, class_idx, layer_name='conv5_block32_concat'):
# # # #     try:
# # # #         image_tensor = tf.convert_to_tensor(image[np.newaxis, ...], dtype=tf.float32)
# # # #         grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])
# # # #         with tf.GradientTape() as tape:
# # # #             conv_outputs, predictions = grad_model(image_tensor)
# # # #             loss = predictions[:, class_idx]
# # # #         grads = tape.gradient(loss, conv_outputs)
# # # #         pooled_grads = tf.reduce_mean(grads, axis=(1, 2))
# # # #         heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1).numpy().squeeze()
# # # #         heatmap = np.maximum(heatmap, 0)
# # # #         if np.max(heatmap) != 0:
# # # #             heatmap /= np.max(heatmap)
# # # #         heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
# # # #         return heatmap
# # # #     except Exception as e:
# # # #         print("Error generating Grad-CAM:", e)
# # # #         return np.zeros((IMG_SIZE, IMG_SIZE))
# # # #
# # # # # LIME
# # # # def lime_explainer(image, model):
# # # #     try:
# # # #         explainer = lime_image.LimeImageExplainer()
# # # #         explanation = explainer.explain_instance(
# # # #             image.astype('double'),
# # # #             model.predict,
# # # #             top_labels=len(CLASSES),
# # # #             hide_color=0,
# # # #             num_samples=1000
# # # #         )
# # # #         pred_class = np.argmax(model.predict(image[np.newaxis, ...]))
# # # #         temp, mask = explanation.get_image_and_mask(pred_class, positive_only=True, num_features=5, hide_rest=False)
# # # #         return mark_boundaries(temp, mask, color=(1, 1, 0))
# # # #     except Exception as e:
# # # #         print("Error generating LIME:", e)
# # # #         return np.zeros_like(image)
# # # #
# # # # # Routes
# # # # @app.route('/')
# # # # def home():
# # # #     return "XAI API for Alzheimer's detection is live!"
# # # #
# # # # @app.route('/predict', methods=['POST'])
# # # # def predict():
# # # #     try:
# # # #         if 'file' not in request.files:
# # # #             return jsonify({"error": "No file uploaded"}), 400
# # # #
# # # #         file = request.files['file']
# # # #         filename = secure_filename(file.filename)
# # # #         file_path = os.path.join(UPLOAD_FOLDER, filename)
# # # #         file.save(file_path)
# # # #
# # # #         img_input, img_raw = preprocess_image(file_path)
# # # #         predictions = model.predict(img_input)[0]
# # # #         pred_index = np.argmax(predictions)
# # # #         pred_label = CLASSES[pred_index]
# # # #         confidence = float(predictions[pred_index] * 100)
# # # #
# # # #         app.config['LAST_IMAGE_PATH'] = file_path
# # # #
# # # #         return jsonify({
# # # #             "actual_label": "Unknown",
# # # #             "predicted_label": pred_label,
# # # #             "confidence": round(confidence, 2)
# # # #         })
# # # #
# # # #     except Exception as e:
# # # #         return jsonify({"error": str(e)}), 500
# # # #
# # # # @app.route('/explain', methods=['GET'])
# # # # def explain():
# # # #     try:
# # # #         if 'LAST_IMAGE_PATH' not in app.config:
# # # #             return jsonify({"error": "No uploaded image yet"}), 400
# # # #
# # # #         file_path = app.config['LAST_IMAGE_PATH']
# # # #         img_input, img_raw = preprocess_image(file_path)
# # # #         pred = model.predict(img_input)[0]
# # # #         pred_class = np.argmax(pred)
# # # #
# # # #         # Grad-CAM
# # # #         gradcam_heatmap = grad_cam(img_raw, model, pred_class)
# # # #         brain_mask = get_brain_mask(img_raw)
# # # #         masked_gradcam = gradcam_heatmap * brain_mask
# # # #         masked_gradcam = np.nan_to_num(masked_gradcam)
# # # #         masked_gradcam = np.clip(masked_gradcam, 0, 1)
# # # #         gradcam_colored = apply_matplotlib_colormap(masked_gradcam)
# # # #         gradcam_overlay = cv2.addWeighted((img_raw * 255).astype(np.uint8), 0.6, gradcam_colored, 0.4, 0)
# # # #         gradcam_encoded = encode_image_to_base64(cv2.cvtColor(gradcam_overlay, cv2.COLOR_BGR2RGB))
# # # #
# # # #         # Integrated Gradients
# # # #         ig_heatmap = integrated_gradients(img_raw, model, pred_class)
# # # #         print("IG Heatmap Stats - Max:", np.max(ig_heatmap), "Min:", np.min(ig_heatmap))
# # # #         if np.max(ig_heatmap) != 0:
# # # #             ig_heatmap = (ig_heatmap - np.min(ig_heatmap)) / (np.max(ig_heatmap) - np.min(ig_heatmap) + 1e-8)
# # # #         ig_colored = apply_matplotlib_colormap(ig_heatmap)
# # # #         ig_overlay = cv2.addWeighted((img_raw * 255).astype(np.uint8), 0.7, ig_colored, 0.3, 0)
# # # #         ig_encoded = encode_image_to_base64(cv2.cvtColor(ig_overlay, cv2.COLOR_BGR2RGB))
# # # #
# # # #         # LIME
# # # #         lime_img = lime_explainer(img_raw, model)
# # # #         lime_encoded = encode_image_to_base64((lime_img * 255).astype(np.uint8))
# # # #
# # # #         return jsonify({
# # # #             "lime": lime_encoded,
# # # #             "gradcam": gradcam_encoded,
# # # #             "ig": ig_encoded
# # # #         })
# # # #
# # # #     except Exception as e:
# # # #         return jsonify({"error": str(e)}), 500
# # # #
# # # # # Run app
# # # # if __name__ == '__main__':
# # # #     app.run(host='0.0.0.0', port=5000, debug=True)
# # #
# # #
# # # # from flask import Flask, request, jsonify
# # # # from flask_cors import CORS
# # # # import os
# # # # import numpy as np
# # # # import tensorflow as tf
# # # # import cv2
# # # # from werkzeug.utils import secure_filename
# # # # from lime import lime_image
# # # # from skimage.segmentation import mark_boundaries
# # # # import base64
# # # # import matplotlib.cm as cm
# # # #
# # # # # Initialize Flask app
# # # # app = Flask(__name__)
# # # # CORS(app)
# # # #
# # # # # Configuration
# # # # MODEL_PATH = "D:/SAI/MRI_DenseNet201_CBAM_best.h5"
# # # # CLASSES = ['AD', 'CN', 'EMCI', 'LMCI']
# # # # IMG_SIZE = 128
# # # # UPLOAD_FOLDER = "uploads"
# # # # os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# # # #
# # # # # Load model
# # # # model = tf.keras.models.load_model(MODEL_PATH)
# # # #
# # # # # Helper to encode image as base64
# # # # def encode_image_to_base64(image):
# # # #     if image.dtype != np.uint8:
# # # #         image = (image * 255).astype(np.uint8)
# # # #     _, buffer = cv2.imencode('.png', image)
# # # #     return base64.b64encode(buffer).decode('utf-8')
# # # #
# # # # # Helper to apply matplotlib colormap (jet)
# # # # def apply_matplotlib_colormap(heatmap):
# # # #     cmap = cm.get_cmap('jet')
# # # #     colored_heatmap = cmap(heatmap)[:, :, :3]
# # # #     return (colored_heatmap * 255).astype(np.uint8)
# # # #
# # # # # Preprocessing
# # # # def preprocess_image(image_path):
# # # #     img = cv2.imread(image_path, cv2.IMREAD_COLOR)
# # # #     if img is None:
# # # #         raise ValueError("Cannot load image")
# # # #     img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
# # # #     img = img.astype(np.float32) / 255.0
# # # #     return np.expand_dims(img, axis=0), img
# # # #
# # # # # Brain mask
# # # # def get_brain_mask(image, threshold=0.1):
# # # #     return (np.mean(image, axis=-1) > threshold).astype(np.float32)
# # # #
# # # # # Integrated Gradients
# # # # def integrated_gradients(image, model, class_idx):
# # # #     image_tensor = tf.convert_to_tensor(image[np.newaxis, ...], dtype=tf.float32)
# # # #     with tf.GradientTape() as tape:
# # # #         tape.watch(image_tensor)
# # # #         preds = model(image_tensor)
# # # #         target = preds[:, class_idx]
# # # #     grads = tape.gradient(target, image_tensor)
# # # #     attribution = np.mean(np.abs(grads.numpy().squeeze()), axis=-1)
# # # #     attribution = np.nan_to_num(attribution)
# # # #     attribution = np.clip(attribution, 0, 1)
# # # #     if np.max(attribution) != 0:
# # # #         attribution = (attribution - np.min(attribution)) / (np.max(attribution) - np.min(attribution) + 1e-8)
# # # #     brain_mask = get_brain_mask(image)
# # # #     return attribution * brain_mask
# # # #
# # # # # Grad-CAM
# # # # def grad_cam(image, model, class_idx, layer_name='conv5_block32_concat'):
# # # #     image_tensor = tf.convert_to_tensor(image[np.newaxis, ...], dtype=tf.float32)
# # # #     grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])
# # # #     with tf.GradientTape() as tape:
# # # #         conv_outputs, predictions = grad_model(image_tensor)
# # # #         loss = predictions[:, class_idx]
# # # #     grads = tape.gradient(loss, conv_outputs)
# # # #     pooled_grads = tf.reduce_mean(grads, axis=(1, 2))
# # # #     heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1).numpy().squeeze()
# # # #     heatmap = np.maximum(heatmap, 0)
# # # #     if np.max(heatmap) != 0:
# # # #         heatmap /= np.max(heatmap)
# # # #     heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
# # # #     return heatmap
# # # #
# # # # # LIME
# # # # def lime_explainer(image, model):
# # # #     explainer = lime_image.LimeImageExplainer()
# # # #     explanation = explainer.explain_instance(
# # # #         image.astype('double'),
# # # #         model.predict,
# # # #         top_labels=len(CLASSES),
# # # #         hide_color=0,
# # # #         num_samples=1000
# # # #     )
# # # #     pred_class = np.argmax(model.predict(image[np.newaxis, ...]))
# # # #     temp, mask = explanation.get_image_and_mask(pred_class, positive_only=True, num_features=5, hide_rest=False)
# # # #     return mark_boundaries(temp, mask, color=(1, 1, 0))
# # # #
# # # # # Routes
# # # # @app.route('/')
# # # # def home():
# # # #     return "XAI API for Alzheimer's detection is live!"
# # # #
# # # # @app.route('/predict', methods=['POST'])
# # # # def predict():
# # # #     try:
# # # #         if 'file' not in request.files:
# # # #             return jsonify({"error": "No file uploaded"}), 400
# # # #
# # # #         file = request.files['file']
# # # #         filename = secure_filename(file.filename)
# # # #         file_path = os.path.join(UPLOAD_FOLDER, filename)
# # # #         file.save(file_path)
# # # #
# # # #         img_input, img_raw = preprocess_image(file_path)
# # # #         predictions = model.predict(img_input)[0]
# # # #         pred_index = np.argmax(predictions)
# # # #         pred_label = CLASSES[pred_index]
# # # #         confidence = float(predictions[pred_index] * 100)
# # # #
# # # #         app.config['LAST_IMAGE_PATH'] = file_path
# # # #
# # # #         return jsonify({
# # # #             "actual_label": "Unknown",
# # # #             "predicted_label": pred_label,
# # # #             "confidence": round(confidence, 2),
# # # #             "all_confidences": {
# # # #                 label: round(float(predictions[i] * 100), 2) for i, label in enumerate(CLASSES)
# # # #             }
# # # #         })
# # # #
# # # #     except Exception as e:
# # # #         return jsonify({"error": str(e)}), 500
# # # #
# # # # @app.route('/explain', methods=['GET'])
# # # # def explain():
# # # #     try:
# # # #         if 'LAST_IMAGE_PATH' not in app.config:
# # # #             return jsonify({"error": "No uploaded image yet"}), 400
# # # #
# # # #         file_path = app.config['LAST_IMAGE_PATH']
# # # #         img_input, img_raw = preprocess_image(file_path)
# # # #         pred = model.predict(img_input)[0]
# # # #         pred_class = np.argmax(pred)
# # # #
# # # #         # Grad-CAM
# # # #         gradcam_heatmap = grad_cam(img_raw, model, pred_class)
# # # #         brain_mask = get_brain_mask(img_raw)
# # # #         masked_gradcam = np.clip(np.nan_to_num(gradcam_heatmap * brain_mask), 0, 1)
# # # #         gradcam_colored = apply_matplotlib_colormap(masked_gradcam)
# # # #         gradcam_overlay = cv2.addWeighted((img_raw * 255).astype(np.uint8), 0.6, gradcam_colored, 0.4, 0)
# # # #         gradcam_encoded = encode_image_to_base64(cv2.cvtColor(gradcam_overlay, cv2.COLOR_BGR2RGB))
# # # #
# # # #         # Integrated Gradients
# # # #         ig_heatmap = integrated_gradients(img_raw, model, pred_class)
# # # #         if np.max(ig_heatmap) != 0:
# # # #             ig_heatmap = (ig_heatmap - np.min(ig_heatmap)) / (np.max(ig_heatmap) - np.min(ig_heatmap) + 1e-8)
# # # #         ig_colored = apply_matplotlib_colormap(ig_heatmap)
# # # #         ig_overlay = cv2.addWeighted((img_raw * 255).astype(np.uint8), 0.7, ig_colored, 0.3, 0)
# # # #         ig_encoded = encode_image_to_base64(cv2.cvtColor(ig_overlay, cv2.COLOR_BGR2RGB))
# # # #
# # # #         # LIME
# # # #         lime_img = lime_explainer(img_raw, model)
# # # #         lime_encoded = encode_image_to_base64((lime_img * 255).astype(np.uint8))
# # # #
# # # #         return jsonify({
# # # #             "lime": lime_encoded,
# # # #             "gradcam": gradcam_encoded,
# # # #             "ig": ig_encoded
# # # #         })
# # # #
# # # #     except Exception as e:
# # # #         return jsonify({"error": str(e)}), 500
# # # #
# # # # # Run app
# # # # if __name__ == '__main__':
# # # #     app.run(host='0.0.0.0', port=5000, debug=True)
# # #
# # #
# # # # from flask import Flask, request, jsonify
# # # # from flask_cors import CORS
# # # # import os
# # # # import numpy as np
# # # # import tensorflow as tf
# # # # import cv2
# # # # from werkzeug.utils import secure_filename
# # # # from lime import lime_image
# # # # from skimage.segmentation import mark_boundaries
# # # # import base64
# # # # import matplotlib.cm as cm
# # # #
# # # # # Initialize Flask app
# # # # app = Flask(__name__)
# # # # CORS(app)
# # # #
# # # # # Configuration
# # # # MODEL_PATH_DENSENET = "D:/SAI/MRI_DenseNet201_CBAM_best.h5"
# # # # MODEL_PATH_RESNET = "D:/SAI/MRI_ResNet50_CBAM_E.h5"
# # # # CLASSES = ['AD', 'CN', 'EMCI', 'LMCI']
# # # # IMG_SIZE = 128
# # # # UPLOAD_FOLDER = "uploads"
# # # # os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# # # #
# # # # # Load models
# # # # densenet_model = tf.keras.models.load_model(MODEL_PATH_DENSENET)
# # # # resnet_model = tf.keras.models.load_model(MODEL_PATH_RESNET)
# # # #
# # # # # Helper functions
# # # # def encode_image_to_base64(image):
# # # #     if image.dtype != np.uint8:
# # # #         image = (image * 255).astype(np.uint8)
# # # #     _, buffer = cv2.imencode('.png', image)
# # # #     return base64.b64encode(buffer).decode('utf-8')
# # # #
# # # # def apply_matplotlib_colormap(heatmap):
# # # #     cmap = cm.get_cmap('jet')
# # # #     colored_heatmap = cmap(heatmap)[:, :, :3]
# # # #     return (colored_heatmap * 255).astype(np.uint8)
# # # #
# # # # def preprocess_image(image_path):
# # # #     img = cv2.imread(image_path, cv2.IMREAD_COLOR)
# # # #     if img is None:
# # # #         raise ValueError("Cannot load image")
# # # #     img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
# # # #     img = img.astype(np.float32) / 255.0
# # # #     return np.expand_dims(img, axis=0), img
# # # #
# # # # def get_brain_mask(image, threshold=0.1):
# # # #     return (np.mean(image, axis=-1) > threshold).astype(np.float32)
# # # #
# # # # def integrated_gradients(image, model, class_idx):
# # # #     image_tensor = tf.convert_to_tensor(image[np.newaxis, ...], dtype=tf.float32)
# # # #     with tf.GradientTape() as tape:
# # # #         tape.watch(image_tensor)
# # # #         preds = model(image_tensor)
# # # #         target = preds[:, class_idx]
# # # #     grads = tape.gradient(target, image_tensor)
# # # #     attribution = np.mean(np.abs(grads.numpy().squeeze()), axis=-1)
# # # #     attribution = np.nan_to_num(attribution)
# # # #     attribution = np.clip(attribution, 0, 1)
# # # #     if np.max(attribution) != 0:
# # # #         attribution = (attribution - np.min(attribution)) / (np.max(attribution) - np.min(attribution) + 1e-8)
# # # #     brain_mask = get_brain_mask(image)
# # # #     return attribution * brain_mask
# # # #
# # # # def grad_cam(image, model, class_idx, layer_name):
# # # #     image_tensor = tf.convert_to_tensor(image[np.newaxis, ...], dtype=tf.float32)
# # # #     grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])
# # # #     with tf.GradientTape() as tape:
# # # #         conv_outputs, predictions = grad_model(image_tensor)
# # # #         loss = predictions[:, class_idx]
# # # #     grads = tape.gradient(loss, conv_outputs)
# # # #     pooled_grads = tf.reduce_mean(grads, axis=(1, 2))
# # # #     heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1).numpy().squeeze()
# # # #     heatmap = np.maximum(heatmap, 0)
# # # #     if np.max(heatmap) != 0:
# # # #         heatmap /= np.max(heatmap)
# # # #     heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
# # # #     return heatmap
# # # #
# # # # def lime_explainer(image, model):
# # # #     explainer = lime_image.LimeImageExplainer()
# # # #     explanation = explainer.explain_instance(
# # # #         image.astype('double'),
# # # #         model.predict,
# # # #         top_labels=len(CLASSES),
# # # #         hide_color=0,
# # # #         num_samples=1000
# # # #     )
# # # #     pred_class = np.argmax(model.predict(image[np.newaxis, ...]))
# # # #     temp, mask = explanation.get_image_and_mask(pred_class, positive_only=True, num_features=5, hide_rest=False)
# # # #     return mark_boundaries(temp, mask, color=(1, 1, 0))
# # # #
# # # # # Routes
# # # # @app.route('/')
# # # # def home():
# # # #     return "XAI Ensemble API for Alzheimer's detection is live!"
# # # #
# # # # @app.route('/predict', methods=['POST'])
# # # # def predict():
# # # #     try:
# # # #         if 'file' not in request.files:
# # # #             return jsonify({"error": "No file uploaded"}), 400
# # # #
# # # #         file = request.files['file']
# # # #         filename = secure_filename(file.filename)
# # # #         file_path = os.path.join(UPLOAD_FOLDER, filename)
# # # #         file.save(file_path)
# # # #
# # # #         img_input, img_raw = preprocess_image(file_path)
# # # #
# # # #         densenet_pred = densenet_model.predict(img_input)[0]
# # # #         resnet_pred = resnet_model.predict(img_input)[0]
# # # #         ensemble_pred = (densenet_pred + resnet_pred) / 2
# # # #
# # # #         pred_index = np.argmax(ensemble_pred)
# # # #         pred_label = CLASSES[pred_index]
# # # #         confidence = float(ensemble_pred[pred_index] * 100)
# # # #
# # # #         app.config['LAST_IMAGE_PATH'] = file_path
# # # #
# # # #         return jsonify({
# # # #             "actual_label": "Unknown",
# # # #             "predicted_label": pred_label,
# # # #             "confidence": round(confidence, 2),
# # # #             "all_confidences": {
# # # #                 label: round(float(ensemble_pred[i] * 100), 2) for i, label in enumerate(CLASSES)
# # # #             }
# # # #         })
# # # #
# # # #     except Exception as e:
# # # #         return jsonify({"error": str(e)}), 500
# # # #
# # # # @app.route('/explain', methods=['GET'])
# # # # def explain():
# # # #     try:
# # # #         if 'LAST_IMAGE_PATH' not in app.config:
# # # #             return jsonify({"error": "No uploaded image yet"}), 400
# # # #
# # # #         file_path = app.config['LAST_IMAGE_PATH']
# # # #         img_input, img_raw = preprocess_image(file_path)
# # # #
# # # #         # Predictions
# # # #         pred_densenet = np.argmax(densenet_model.predict(img_input)[0])
# # # #         pred_resnet = np.argmax(resnet_model.predict(img_input)[0])
# # # #
# # # #         # DenseNet explanations
# # # #         gradcam_dense = grad_cam(img_raw, densenet_model, pred_densenet, 'conv5_block32_concat')
# # # #         ig_dense = integrated_gradients(img_raw, densenet_model, pred_densenet)
# # # #         lime_dense = lime_explainer(img_raw, densenet_model)
# # # #
# # # #         # ResNet explanations
# # # #         gradcam_resnet = grad_cam(img_raw, resnet_model, pred_resnet, 'conv5_block3_out')
# # # #         ig_resnet = integrated_gradients(img_raw, resnet_model, pred_resnet)
# # # #         lime_resnet = lime_explainer(img_raw, resnet_model)
# # # #
# # # #         # Apply visual formatting
# # # #         def format_explanation(img_raw, heatmap):
# # # #             heatmap = np.nan_to_num(np.clip(heatmap, 0, 1))
# # # #             colored = apply_matplotlib_colormap(heatmap)
# # # #             overlay = cv2.addWeighted((img_raw * 255).astype(np.uint8), 0.6, colored, 0.4, 0)
# # # #             return encode_image_to_base64(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
# # # #
# # # #         return jsonify({
# # # #             "gradcam": {
# # # #                 "densenet": format_explanation(img_raw, gradcam_dense),
# # # #                 "resnet": format_explanation(img_raw, gradcam_resnet)
# # # #             },
# # # #             "ig": {
# # # #                 "densenet": format_explanation(img_raw, ig_dense),
# # # #                 "resnet": format_explanation(img_raw, ig_resnet)
# # # #             },
# # # #             "lime": {
# # # #                 "densenet": encode_image_to_base64((lime_dense * 255).astype(np.uint8)),
# # # #                 "resnet": encode_image_to_base64((lime_resnet * 255).astype(np.uint8))
# # # #             }
# # # #         })
# # # #
# # # #     except Exception as e:
# # # #         return jsonify({"error": str(e)}), 500
# # # #
# # # # # Run app
# # # # if __name__ == '__main__':
# # # #     app.run(host='0.0.0.0', port=5000, debug=True)
# # #
# # # from flask import Flask, request, jsonify
# # # from flask_cors import CORS
# # # import os
# # # import numpy as np
# # # import tensorflow as tf
# # # import cv2
# # # from werkzeug.utils import secure_filename
# # # from lime import lime_image
# # # from skimage.segmentation import mark_boundaries
# # # import base64
# # # import matplotlib.cm as cm
# # #
# # # # Initialize Flask app
# # # app = Flask(__name__)
# # # CORS(app)
# # #
# # # # Configuration
# # # MODEL_PATH_DENSENET = "D:/SAI/MRI_DenseNet201_CBAM_best.h5"
# # # MODEL_PATH_RESNET = "D:/SAI/MRI_ResNet50_CBAM_E.h5"
# # # CLASSES = ['AD', 'CN', 'EMCI', 'LMCI']
# # # IMG_SIZE = 128
# # # UPLOAD_FOLDER = "uploads"
# # # os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# # #
# # # # Load models
# # # densenet_model = tf.keras.models.load_model(MODEL_PATH_DENSENET)
# # # resnet_model = tf.keras.models.load_model(MODEL_PATH_RESNET)
# # #
# # # # Helper functions
# # # def encode_image_to_base64(image):
# # #     if image.dtype != np.uint8:
# # #         image = (image * 255).astype(np.uint8)
# # #     _, buffer = cv2.imencode('.png', image)
# # #     return base64.b64encode(buffer).decode('utf-8')
# # #
# # # def apply_matplotlib_colormap(heatmap):
# # #     cmap = cm.get_cmap('jet')
# # #     colored_heatmap = cmap(heatmap)[:, :, :3]
# # #     return (colored_heatmap * 255).astype(np.uint8)
# # #
# # # def preprocess_image(image_path):
# # #     img = cv2.imread(image_path, cv2.IMREAD_COLOR)
# # #     if img is None:
# # #         raise ValueError("Cannot load image")
# # #     img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
# # #     img = img.astype(np.float32) / 255.0
# # #     return np.expand_dims(img, axis=0), img
# # #
# # # def get_brain_mask(image, threshold=0.1):
# # #     return (np.mean(image, axis=-1) > threshold).astype(np.float32)
# # #
# # # def integrated_gradients(image, model, class_idx):
# # #     image_tensor = tf.convert_to_tensor(image[np.newaxis, ...], dtype=tf.float32)
# # #     with tf.GradientTape() as tape:
# # #         tape.watch(image_tensor)
# # #         preds = model(image_tensor)
# # #         target = preds[:, class_idx]
# # #     grads = tape.gradient(target, image_tensor)
# # #     attribution = np.mean(np.abs(grads.numpy().squeeze()), axis=-1)
# # #     attribution = np.nan_to_num(attribution)
# # #     attribution = np.clip(attribution, 0, 1)
# # #     if np.max(attribution) != 0:
# # #         attribution = (attribution - np.min(attribution)) / (np.max(attribution) - np.min(attribution) + 1e-8)
# # #     brain_mask = get_brain_mask(image)
# # #     return attribution * brain_mask
# # #
# # # def grad_cam(image, model, class_idx, layer_name):
# # #     image_tensor = tf.convert_to_tensor(image[np.newaxis, ...], dtype=tf.float32)
# # #     grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])
# # #     with tf.GradientTape() as tape:
# # #         conv_outputs, predictions = grad_model(image_tensor)
# # #         loss = predictions[:, class_idx]
# # #     grads = tape.gradient(loss, conv_outputs)
# # #     pooled_grads = tf.reduce_mean(grads, axis=(1, 2))
# # #     heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1).numpy().squeeze()
# # #     heatmap = np.maximum(heatmap, 0)
# # #     if np.max(heatmap) != 0:
# # #         heatmap /= np.max(heatmap)
# # #     heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
# # #     return heatmap
# # #
# # # def lime_explainer(image, model):
# # #     explainer = lime_image.LimeImageExplainer()
# # #     explanation = explainer.explain_instance(
# # #         image.astype('double'),
# # #         model.predict,
# # #         top_labels=len(CLASSES),
# # #         hide_color=0,
# # #         num_samples=1000
# # #     )
# # #     pred_class = np.argmax(model.predict(image[np.newaxis, ...]))
# # #     temp, mask = explanation.get_image_and_mask(pred_class, positive_only=True, num_features=5, hide_rest=False)
# # #     return mark_boundaries(temp, mask, color=(1, 1, 0))
# # #
# # # # Routes
# # # @app.route('/')
# # # def home():
# # #     return "XAI Ensemble API for Alzheimer's detection is live!"
# # #
# # # @app.route('/predict_mri', methods=['POST'])
# # # def predict():
# # #     try:
# # #         if 'file' not in request.files:
# # #             return jsonify({"error": "No file uploaded"}), 400
# # #
# # #         file = request.files['file']
# # #         filename = secure_filename(file.filename)
# # #         file_path = os.path.join(UPLOAD_FOLDER, filename)
# # #         file.save(file_path)
# # #
# # #         img_input, img_raw = preprocess_image(file_path)
# # #
# # #         densenet_pred = densenet_model.predict(img_input)[0]
# # #         resnet_pred = resnet_model.predict(img_input)[0]
# # #         ensemble_pred = (densenet_pred + resnet_pred) / 2
# # #
# # #         pred_index = np.argmax(ensemble_pred)
# # #         pred_label = CLASSES[pred_index]
# # #         confidence = float(ensemble_pred[pred_index] * 100)
# # #
# # #         app.config['LAST_IMAGE_PATH'] = file_path
# # #
# # #         return jsonify({
# # #             "actual_label": "Unknown",
# # #             "predicted_label": pred_label,
# # #             "confidence": round(confidence, 2),
# # #             "all_confidences": {
# # #                 label: round(float(ensemble_pred[i] * 100), 2) for i, label in enumerate(CLASSES)
# # #             }
# # #         })
# # #
# # #     except Exception as e:
# # #         return jsonify({"error": str(e)}), 500
# # #
# # # @app.route('/explain', methods=['GET'])
# # # def explain():
# # #     try:
# # #         if 'LAST_IMAGE_PATH' not in app.config:
# # #             return jsonify({"error": "No uploaded image yet"}), 400
# # #
# # #         file_path = app.config['LAST_IMAGE_PATH']
# # #         img_input, img_raw = preprocess_image(file_path)
# # #
# # #         pred_densenet = np.argmax(densenet_model.predict(img_input)[0])
# # #         pred_resnet = np.argmax(resnet_model.predict(img_input)[0])
# # #
# # #         # Grad-CAM
# # #         gradcam_dense = grad_cam(img_raw, densenet_model, pred_densenet, 'conv5_block32_concat')
# # #         brain_mask = get_brain_mask(img_raw)
# # #         gradcam_dense = np.clip(np.nan_to_num(gradcam_dense * brain_mask), 0, 1)
# # #         gradcam_dense_overlay = apply_matplotlib_colormap(gradcam_dense)
# # #         gradcam_dense_overlay = cv2.addWeighted((img_raw * 255).astype(np.uint8), 0.6, gradcam_dense_overlay, 0.4, 0)
# # #
# # #         gradcam_resnet = grad_cam(img_raw, resnet_model, pred_resnet, 'conv5_block3_out')
# # #         gradcam_resnet = np.clip(np.nan_to_num(gradcam_resnet * brain_mask), 0, 1)
# # #         gradcam_resnet_overlay = apply_matplotlib_colormap(gradcam_resnet)
# # #         gradcam_resnet_overlay = cv2.addWeighted((img_raw * 255).astype(np.uint8), 0.6, gradcam_resnet_overlay, 0.4, 0)
# # #
# # #         # Integrated Gradients
# # #         ig_dense = integrated_gradients(img_raw, densenet_model, pred_densenet)
# # #         if np.max(ig_dense) != 0:
# # #             ig_dense = (ig_dense - np.min(ig_dense)) / (np.max(ig_dense) - np.min(ig_dense) + 1e-8)
# # #         ig_dense_overlay = apply_matplotlib_colormap(ig_dense)
# # #         ig_dense_overlay = cv2.addWeighted((img_raw * 255).astype(np.uint8), 0.7, ig_dense_overlay, 0.3, 0)
# # #
# # #         ig_resnet = integrated_gradients(img_raw, resnet_model, pred_resnet)
# # #         if np.max(ig_resnet) != 0:
# # #             ig_resnet = (ig_resnet - np.min(ig_resnet)) / (np.max(ig_resnet) - np.min(ig_resnet) + 1e-8)
# # #         ig_resnet_overlay = apply_matplotlib_colormap(ig_resnet)
# # #         ig_resnet_overlay = cv2.addWeighted((img_raw * 255).astype(np.uint8), 0.7, ig_resnet_overlay, 0.3, 0)
# # #
# # #         # LIME
# # #         lime_dense = lime_explainer(img_raw, densenet_model)
# # #         lime_resnet = lime_explainer(img_raw, resnet_model)
# # #
# # #         return jsonify({
# # #             "lime": {
# # #                 "densenet": encode_image_to_base64((lime_dense * 255).astype(np.uint8)),
# # #                 "resnet": encode_image_to_base64((lime_resnet * 255).astype(np.uint8))
# # #             },
# # #             "gradcam": {
# # #                 "densenet": encode_image_to_base64(cv2.cvtColor(gradcam_dense_overlay, cv2.COLOR_BGR2RGB)),
# # #                 "resnet": encode_image_to_base64(cv2.cvtColor(gradcam_resnet_overlay, cv2.COLOR_BGR2RGB))
# # #             },
# # #             "ig": {
# # #                 "densenet": encode_image_to_base64(cv2.cvtColor(ig_dense_overlay, cv2.COLOR_BGR2RGB)),
# # #                 "resnet": encode_image_to_base64(cv2.cvtColor(ig_resnet_overlay, cv2.COLOR_BGR2RGB))
# # #             }
# # #         })
# # #
# # #     except Exception as e:
# # #         return jsonify({"error": str(e)}), 500
# # #
# # # # Run app
# # # if __name__ == '__main__':
# # #     app.run(host='0.0.0.0', port=5000, debug=True)
# #
# # from flask import Flask, request, jsonify
# # from flask_cors import CORS
# # import os
# # import numpy as np
# # import tensorflow as tf
# # import cv2
# # from werkzeug.utils import secure_filename
# # from lime import lime_image
# # from skimage.segmentation import mark_boundaries
# # import base64
# # import matplotlib.cm as cm
# #
# # # Initialize Flask app
# # app = Flask(__name__)
# # CORS(app)
# #
# # # Configuration
# # MRI_MODEL_PATH_DENSENET = "D:/SAI/MRI_DenseNet201_CBAM_best.h5"
# # MRI_MODEL_PATH_RESNET = "D:/SAI/MRI_ResNet50_CBAM_E.h5"
# # PET_MODEL_PATH = "D:/SAI/PET_VGG19_CBAM_Enhanced.h5"
# # CLASSES = ['AD', 'CN', 'EMCI', 'LMCI']
# # IMG_SIZE = 128
# # UPLOAD_FOLDER = "uploads"
# # os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# #
# # # Load models
# # mri_densenet_model = tf.keras.models.load_model(MRI_MODEL_PATH_DENSENET)
# # mri_resnet_model = tf.keras.models.load_model(MRI_MODEL_PATH_RESNET)
# # pet_model = tf.keras.models.load_model(PET_MODEL_PATH)
# #
# # # Helpers
# # def encode_image_to_base64(image):
# #     if image.dtype != np.uint8:
# #         image = (image * 255).astype(np.uint8)
# #     _, buffer = cv2.imencode('.png', image)
# #     return base64.b64encode(buffer).decode('utf-8')
# #
# # def apply_matplotlib_colormap(heatmap):
# #     cmap = cm.get_cmap('jet')
# #     colored_heatmap = cmap(heatmap)[:, :, :3]
# #     return (colored_heatmap * 255).astype(np.uint8)
# #
# # def preprocess_image(image_path):
# #     img = cv2.imread(image_path, cv2.IMREAD_COLOR)
# #     if img is None:
# #         raise ValueError("Cannot load image")
# #     img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
# #     img = img.astype(np.float32) / 255.0
# #     return np.expand_dims(img, axis=0), img
# #
# # def get_brain_mask(image, threshold=0.1):
# #     return (np.mean(image, axis=-1) > threshold).astype(np.float32)
# #
# # def integrated_gradients(image, model, class_idx):
# #     image_tensor = tf.convert_to_tensor(image[np.newaxis, ...], dtype=tf.float32)
# #     with tf.GradientTape() as tape:
# #         tape.watch(image_tensor)
# #         preds = model(image_tensor)
# #         target = preds[:, class_idx]
# #     grads = tape.gradient(target, image_tensor)
# #     attribution = np.mean(np.abs(grads.numpy().squeeze()), axis=-1)
# #     attribution = np.nan_to_num(attribution)
# #     attribution = np.clip(attribution, 0, 1)
# #     if np.max(attribution) != 0:
# #         attribution = (attribution - np.min(attribution)) / (np.max(attribution) - np.min(attribution) + 1e-8)
# #     brain_mask = get_brain_mask(image)
# #     return attribution * brain_mask
# #
# # def grad_cam(image, model, class_idx, layer_name):
# #     image_tensor = tf.convert_to_tensor(image[np.newaxis, ...], dtype=tf.float32)
# #     grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])
# #     with tf.GradientTape() as tape:
# #         conv_outputs, predictions = grad_model(image_tensor)
# #         loss = predictions[:, class_idx]
# #     grads = tape.gradient(loss, conv_outputs)
# #     pooled_grads = tf.reduce_mean(grads, axis=(1, 2))
# #     heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1).numpy().squeeze()
# #     heatmap = np.maximum(heatmap, 0)
# #     if np.max(heatmap) != 0:
# #         heatmap /= np.max(heatmap)
# #     heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
# #     return heatmap
# #
# # def lime_explainer(image, model):
# #     explainer = lime_image.LimeImageExplainer()
# #     explanation = explainer.explain_instance(
# #         image.astype('double'),
# #         model.predict,
# #         top_labels=len(CLASSES),
# #         hide_color=0,
# #         num_samples=1000
# #     )
# #     pred_class = np.argmax(model.predict(image[np.newaxis, ...]))
# #     temp, mask = explanation.get_image_and_mask(pred_class, positive_only=True, num_features=5, hide_rest=False)
# #     return mark_boundaries(temp, mask, color=(1, 1, 0))
# #
# # # Prediction route for PET
# # @app.route('/predict_pet', methods=['POST'])
# # def predict_pet():
# #     try:
# #         if 'file' not in request.files:
# #             return jsonify({"error": "No file uploaded"}), 400
# #
# #         file = request.files['file']
# #         filename = secure_filename(file.filename)
# #         file_path = os.path.join(UPLOAD_FOLDER, filename)
# #         file.save(file_path)
# #
# #         img_input, _ = preprocess_image(file_path)
# #         pred = pet_model.predict(img_input)[0]
# #
# #         pred_index = np.argmax(pred)
# #         pred_label = CLASSES[pred_index]
# #         confidence = float(pred[pred_index] * 100)
# #
# #         app.config['LAST_IMAGE_PATH'] = file_path
# #         app.config['LAST_MODEL_TYPE'] = 'pet'
# #
# #         return jsonify({
# #             "actual_label": "Unknown",
# #             "predicted_label": pred_label,
# #             "confidence": round(confidence, 2),
# #             "all_confidences": {
# #                 label: round(float(pred[i] * 100), 2) for i, label in enumerate(CLASSES)
# #             }
# #         })
# #
# #     except Exception as e:
# #         return jsonify({"error": str(e)}), 500
# #
# # @app.route('/explain_pet', methods=['GET'])
# # def explain_pet():
# #     try:
# #         if 'LAST_IMAGE_PATH' not in app.config:
# #             return jsonify({"error": "No uploaded image yet"}), 400
# #
# #         file_path = app.config['LAST_IMAGE_PATH']
# #         img_input, img_raw = preprocess_image(file_path)
# #         pred_class = np.argmax(pet_model.predict(img_input)[0])
# #
# #         # Grad-CAM
# #         gradcam = grad_cam(img_raw, pet_model, pred_class, 'block5_conv4')
# #         gradcam = np.clip(np.nan_to_num(gradcam * get_brain_mask(img_raw)), 0, 1)
# #         gradcam_overlay = apply_matplotlib_colormap(gradcam)
# #         gradcam_overlay = cv2.addWeighted((img_raw * 255).astype(np.uint8), 0.6, gradcam_overlay, 0.4, 0)
# #
# #         # IG
# #         ig = integrated_gradients(img_raw, pet_model, pred_class)
# #         if np.max(ig) != 0:
# #             ig = (ig - np.min(ig)) / (np.max(ig) - np.min(ig) + 1e-8)
# #         ig_overlay = apply_matplotlib_colormap(ig)
# #         ig_overlay = cv2.addWeighted((img_raw * 255).astype(np.uint8), 0.7, ig_overlay, 0.3, 0)
# #
# #         # LIME
# #         lime_img = lime_explainer(img_raw, pet_model)
# #
# #         return jsonify({
# #             "lime": encode_image_to_base64((lime_img * 255).astype(np.uint8)),
# #             "gradcam": encode_image_to_base64(cv2.cvtColor(gradcam_overlay, cv2.COLOR_BGR2RGB)),
# #             "ig": encode_image_to_base64(cv2.cvtColor(ig_overlay, cv2.COLOR_BGR2RGB))
# #         })
# #
# #     except Exception as e:
# #         return jsonify({"error": str(e)}), 500
# #
# # # Run app
# # if __name__ == '__main__':
# #     app.run(host='0.0.0.0', port=5000, debug=True)
#
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import os
# import numpy as np
# import tensorflow as tf
# import cv2
# from werkzeug.utils import secure_filename
# from lime import lime_image
# from skimage.segmentation import mark_boundaries
# import base64
# import matplotlib.cm as cm
#
# # Initialize Flask app
# app = Flask(__name__)
# CORS(app)
#
# # Configuration
# MRI_MODEL_PATH_DENSENET = "D:/SAI/MRI_DenseNet201_CBAM_best.h5"
# MRI_MODEL_PATH_RESNET = "D:/SAI/MRI_ResNet50_CBAM_E.h5"
# PET_MODEL_PATH = "D:/SAI/PET_VGG19_CBAM_Enhanced.h5"
# CLASSES = ['AD', 'CN', 'EMCI', 'LMCI']
# IMG_SIZE = 128
# UPLOAD_FOLDER = "uploads"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
#
# # Load models
# mri_densenet_model = tf.keras.models.load_model(MRI_MODEL_PATH_DENSENET)
# mri_resnet_model = tf.keras.models.load_model(MRI_MODEL_PATH_RESNET)
# pet_model = tf.keras.models.load_model(PET_MODEL_PATH)
#
# # Helpers
# def encode_image_to_base64(image):
#     if image.dtype != np.uint8:
#         image = (image * 255).astype(np.uint8)
#     _, buffer = cv2.imencode('.png', image)
#     return base64.b64encode(buffer).decode('utf-8')
#
# def apply_matplotlib_colormap(heatmap):
#     cmap = cm.get_cmap('jet')
#     colored_heatmap = cmap(heatmap)[:, :, :3]
#     return (colored_heatmap * 255).astype(np.uint8)
#
# def preprocess_image(image_path):
#     img = cv2.imread(image_path, cv2.IMREAD_COLOR)
#     if img is None:
#         raise ValueError("Cannot load image")
#     img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
#     img = img.astype(np.float32) / 255.0
#     return np.expand_dims(img, axis=0), img
#
# def get_brain_mask(image, threshold=0.1):
#     return (np.mean(image, axis=-1) > threshold).astype(np.float32)
#
# def integrated_gradients(image, model, class_idx):
#     image_tensor = tf.convert_to_tensor(image[np.newaxis, ...], dtype=tf.float32)
#     with tf.GradientTape() as tape:
#         tape.watch(image_tensor)
#         preds = model(image_tensor)
#         target = preds[:, class_idx]
#     grads = tape.gradient(target, image_tensor)
#     attribution = np.mean(np.abs(grads.numpy().squeeze()), axis=-1)
#     attribution = np.nan_to_num(attribution)
#     attribution = np.clip(attribution, 0, 1)
#     if np.max(attribution) != 0:
#         attribution = (attribution - np.min(attribution)) / (np.max(attribution) - np.min(attribution) + 1e-8)
#     brain_mask = get_brain_mask(image)
#     return attribution * brain_mask
#
# def grad_cam(image, model, class_idx, layer_name):
#     image_tensor = tf.convert_to_tensor(image[np.newaxis, ...], dtype=tf.float32)
#     grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])
#     with tf.GradientTape() as tape:
#         conv_outputs, predictions = grad_model(image_tensor)
#         loss = predictions[:, class_idx]
#     grads = tape.gradient(loss, conv_outputs)
#     pooled_grads = tf.reduce_mean(grads, axis=(1, 2))
#     heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1).numpy().squeeze()
#     heatmap = np.maximum(heatmap, 0)
#     if np.max(heatmap) != 0:
#         heatmap /= np.max(heatmap)
#     heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
#     return heatmap
#
# def lime_explainer(image, model):
#     explainer = lime_image.LimeImageExplainer()
#     explanation = explainer.explain_instance(
#         image.astype('double'),
#         model.predict,
#         top_labels=len(CLASSES),
#         hide_color=0,
#         num_samples=1000
#     )
#     pred_class = np.argmax(model.predict(image[np.newaxis, ...]))
#     temp, mask = explanation.get_image_and_mask(pred_class, positive_only=True, num_features=5, hide_rest=False)
#     return mark_boundaries(temp, mask, color=(1, 1, 0))
#
# @app.route('/')
# def home():
#     return "XAI API for Alzheimer's detection is live!"
#
# # MRI Prediction
# @app.route('/predict_mri', methods=['POST'])
# def predict_mri():
#     try:
#         if 'file' not in request.files:
#             return jsonify({"error": "No file uploaded"}), 400
#
#         file = request.files['file']
#         filename = secure_filename(file.filename)
#         file_path = os.path.join(UPLOAD_FOLDER, filename)
#         file.save(file_path)
#
#         img_input, img_raw = preprocess_image(file_path)
#
#         densenet_pred = mri_densenet_model.predict(img_input)[0]
#         resnet_pred = mri_resnet_model.predict(img_input)[0]
#         ensemble_pred = (densenet_pred + resnet_pred) / 2
#
#         pred_index = np.argmax(ensemble_pred)
#         pred_label = CLASSES[pred_index]
#         confidence = float(ensemble_pred[pred_index] * 100)
#
#         app.config['LAST_IMAGE_PATH'] = file_path
#
#         return jsonify({
#             "actual_label": "Unknown",
#             "predicted_label": pred_label,
#             "confidence": round(confidence, 2),
#             "all_confidences": {
#                 label: round(float(ensemble_pred[i] * 100), 2) for i, label in enumerate(CLASSES)
#             }
#         })
#
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
#
# # PET Prediction
# @app.route('/predict_pet', methods=['POST'])
# def predict_pet():
#     try:
#         if 'file' not in request.files:
#             return jsonify({"error": "No file uploaded"}), 400
#
#         file = request.files['file']
#         filename = secure_filename(file.filename)
#         file_path = os.path.join(UPLOAD_FOLDER, filename)
#         file.save(file_path)
#
#         img_input, _ = preprocess_image(file_path)
#         pred = pet_model.predict(img_input)[0]
#
#         pred_index = np.argmax(pred)
#         pred_label = CLASSES[pred_index]
#         confidence = float(pred[pred_index] * 100)
#
#         app.config['LAST_IMAGE_PATH'] = file_path
#         app.config['LAST_MODEL_TYPE'] = 'pet'
#
#         return jsonify({
#             "actual_label": "Unknown",
#             "predicted_label": pred_label,
#             "confidence": round(confidence, 2),
#             "all_confidences": {
#                 label: round(float(pred[i] * 100), 2) for i, label in enumerate(CLASSES)
#             }
#         })
#
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
#
# @app.route('/explain_pet', methods=['GET'])
# def explain_pet():
#     try:
#         if 'LAST_IMAGE_PATH' not in app.config:
#             return jsonify({"error": "No uploaded image yet"}), 400
#
#         file_path = app.config['LAST_IMAGE_PATH']
#         img_input, img_raw = preprocess_image(file_path)
#         pred_class = np.argmax(pet_model.predict(img_input)[0])
#
#         gradcam = grad_cam(img_raw, pet_model, pred_class, 'block5_conv4')
#         gradcam = np.clip(np.nan_to_num(gradcam * get_brain_mask(img_raw)), 0, 1)
#         gradcam_overlay = apply_matplotlib_colormap(gradcam)
#         gradcam_overlay = cv2.addWeighted((img_raw * 255).astype(np.uint8), 0.6, gradcam_overlay, 0.4, 0)
#
#         ig = integrated_gradients(img_raw, pet_model, pred_class)
#         if np.max(ig) != 0:
#             ig = (ig - np.min(ig)) / (np.max(ig) - np.min(ig) + 1e-8)
#         ig_overlay = apply_matplotlib_colormap(ig)
#         ig_overlay = cv2.addWeighted((img_raw * 255).astype(np.uint8), 0.7, ig_overlay, 0.3, 0)
#
#         lime_img = lime_explainer(img_raw, pet_model)
#
#         return jsonify({
#             "lime": encode_image_to_base64((lime_img * 255).astype(np.uint8)),
#             "gradcam": encode_image_to_base64(cv2.cvtColor(gradcam_overlay, cv2.COLOR_BGR2RGB)),
#             "ig": encode_image_to_base64(cv2.cvtColor(ig_overlay, cv2.COLOR_BGR2RGB))
#         })
#
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
#
# # Run app
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=True)


# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import os
# import numpy as np
# import tensorflow as tf
# import cv2
# from werkzeug.utils import secure_filename
# from lime import lime_image
# from skimage.segmentation import mark_boundaries
# import base64
# import matplotlib.cm as cm
#
# app = Flask(__name__)
# CORS(app)
#
# # Paths
# MRI_MODEL_PATH_DENSENET = "D:/SAI/MRI_DenseNet201_CBAM_best.h5"
# MRI_MODEL_PATH_RESNET = "D:/SAI/MRI_ResNet50_CBAM_E.h5"
# PET_MODEL_PATH = "D:/SAI/PET_VGG19_CBAM_Enhanced.h5"
# UPLOAD_FOLDER = "uploads"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
#
# # Constants
# IMG_SIZE = 128
# CLASSES = ['AD', 'CN', 'EMCI', 'LMCI']
#
# # Load Models
# mri_densenet = tf.keras.models.load_model(MRI_MODEL_PATH_DENSENET)
# mri_resnet = tf.keras.models.load_model(MRI_MODEL_PATH_RESNET)
# pet_model = tf.keras.models.load_model(PET_MODEL_PATH)
#
# # Helpers
# def encode_image_to_base64(image):
#     if image.dtype != np.uint8:
#         image = (image * 255).astype(np.uint8)
#     _, buffer = cv2.imencode('.png', image)
#     return base64.b64encode(buffer).decode('utf-8')
#
# def apply_matplotlib_colormap(heatmap):
#     cmap = cm.get_cmap('jet')
#     colored_heatmap = cmap(heatmap)[:, :, :3]
#     return (colored_heatmap * 255).astype(np.uint8)
#
# def preprocess_image(image_path):
#     img = cv2.imread(image_path, cv2.IMREAD_COLOR)
#     if img is None:
#         raise ValueError("Cannot load image")
#     img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
#     img = img.astype(np.float32) / 255.0
#     return np.expand_dims(img, axis=0), img
#
# def get_brain_mask(image, threshold=0.1):
#     return (np.mean(image, axis=-1) > threshold).astype(np.float32)
# #
# # def integrated_gradients(image, model, class_idx):
# #     image_tensor = tf.convert_to_tensor(image[np.newaxis, ...], dtype=tf.float32)
# #     with tf.GradientTape() as tape:
# #         tape.watch(image_tensor)
# #         preds = model(image_tensor)
# #         target = preds[:, class_idx]
# #     grads = tape.gradient(target, image_tensor)
# #     attribution = np.mean(np.abs(grads.numpy().squeeze()), axis=-1)
# #     attribution = np.nan_to_num(attribution)
# #     attribution = np.clip(attribution, 0, 1)
# #     if np.max(attribution) != 0:
# #         attribution = (attribution - np.min(attribution)) / (np.max(attribution) - np.min(attribution) + 1e-8)
# #     return attribution * get_brain_mask(image)
#
# def integrated_gradients(image, model, class_index, steps=50):
#     image = image.astype(np.float32)
#     baseline = np.zeros_like(image).astype(np.float32)  # Black image baseline
#
#     # Scale inputs between baseline and input image
#     interpolated_images = [baseline + (float(i) / steps) * (image - baseline) for i in range(steps + 1)]
#     interpolated_images = tf.convert_to_tensor(np.array(interpolated_images), dtype=tf.float32)
#
#     with tf.GradientTape() as tape:
#         tape.watch(interpolated_images)
#         preds = model(interpolated_images)
#         target = preds[:, class_index]
#
#     grads = tape.gradient(target, interpolated_images).numpy()
#     avg_grads = np.mean(grads, axis=0)  # Average across steps
#     integrated_grads = (image - baseline) * avg_grads
#
#     # Aggregate across channels
#     attributions = np.mean(np.abs(integrated_grads), axis=-1)
#     attributions = np.clip(attributions, 0, 1)
#
#     if np.max(attributions) > 0:
#         attributions = (attributions - np.min(attributions)) / (np.max(attributions) - np.min(attributions) + 1e-8)
#
#     # Mask out background
#     return attributions * get_brain_mask(image)
#
#
# def grad_cam(image, model, class_idx, layer_name):
#     image_tensor = tf.convert_to_tensor(image[np.newaxis, ...], dtype=tf.float32)
#     grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])
#     with tf.GradientTape() as tape:
#         conv_outputs, predictions = grad_model(image_tensor)
#         loss = predictions[:, class_idx]
#     grads = tape.gradient(loss, conv_outputs)
#     pooled_grads = tf.reduce_mean(grads, axis=(1, 2))
#     heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1).numpy().squeeze()
#     heatmap = np.maximum(heatmap, 0)
#     if np.max(heatmap) != 0:
#         heatmap /= np.max(heatmap)
#     return cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
#
# # def lime_explainer(image, model):
# #     explainer = lime_image.LimeImageExplainer()
# #     explanation = explainer.explain_instance(
# #         image.astype('double'),
# #         model.predict,
# #         top_labels=len(CLASSES),
# #         hide_color=0,
# #         num_samples=1000
# #     )
# #     pred_class = np.argmax(model.predict(image[np.newaxis, ...]))
# #     temp, mask = explanation.get_image_and_mask(pred_class, positive_only=True, num_features=5, hide_rest=False)
# #     return mark_boundaries(temp, mask, color=(1, 1, 0))
#
# def lime_explainer(image, model):
#     explainer = lime_image.LimeImageExplainer()
#     explanation = explainer.explain_instance(
#         image.astype('double'),
#         model.predict,
#         top_labels=len(CLASSES),
#         hide_color=0,        # You can also try None
#         num_samples=500
#     )
#
#     pred_class = np.argmax(model.predict(image[np.newaxis, ...]))
#     temp, mask = explanation.get_image_and_mask(
#         pred_class,
#         positive_only=True,
#         num_features=8,       # Try increasing number of features
#         hide_rest=False
#     )
#
#     # Make temp uint8 for better visualization
#     temp_uint8 = (temp * 255).astype(np.uint8)
#     boundaries = mark_boundaries(temp_uint8, mask, color=(1, 1, 0), mode='thick')
#
#     return boundaries
#
#
# # --- Routes ---
#
# @app.route('/')
# def home():
#     return "XAI MRI & PET API for Alzheimer's detection is live!"
#
# @app.route('/predict_mri', methods=['POST'])
# def predict_mri():
#     try:
#         file = request.files['file']
#         path = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
#         file.save(path)
#
#         img_input, _ = preprocess_image(path)
#
#         pred1 = mri_densenet.predict(img_input)[0]
#         pred2 = mri_resnet.predict(img_input)[0]
#         ensemble = (pred1 + pred2) / 2
#         pred_idx = np.argmax(ensemble)
#
#         app.config['MRI_PATH'] = path
#         return jsonify({
#             "predicted_label": CLASSES[pred_idx],
#             "confidence": round(float(ensemble[pred_idx]) * 100, 2),
#             "actual_label": "Unknown",
#             "all_confidences": {CLASSES[i]: round(float(ensemble[i] * 100), 2) for i in range(len(CLASSES))}
#         })
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
#
#
# @app.route('/explain_mri', methods=['GET'])
# def explain_mri():
#     try:
#         path = app.config['MRI_PATH']
#         img_input, img_raw = preprocess_image(path)
#
#         pred1 = np.argmax(mri_densenet.predict(img_input)[0])
#         pred2 = np.argmax(mri_resnet.predict(img_input)[0])
#
#         def process(model, pred_class, layer_name):
#             grad = grad_cam(img_raw, model, pred_class, layer_name)
#             grad_overlay = apply_matplotlib_colormap(np.clip(grad * get_brain_mask(img_raw), 0, 1))
#             grad_overlay = cv2.addWeighted((img_raw * 255).astype(np.uint8), 0.6, grad_overlay, 0.4, 0)
#
#             # ig = integrated_gradients(img_raw, model, pred_class)
#             # ig_overlay = apply_matplotlib_colormap(ig)
#             # ig_overlay = cv2.addWeighted((img_raw * 255).astype(np.uint8), 0.7, ig_overlay, 0.3, 0)
#
#             ig = integrated_gradients(img_raw, model, pred_class)
#
#             # Apply gamma correction to boost weak signals
#             ig = ig ** 0.5  # You can also try 0.4 or 0.6 depending on visibility
#
#             ig_overlay = apply_matplotlib_colormap(ig)
#
#             # Stronger overlay (more heatmap visibility)
#             ig_overlay = cv2.addWeighted((img_raw * 255).astype(np.uint8), 0.4, ig_overlay, 0.6, 0)
#
#             lime_img = lime_explainer(img_raw, model)
#
#             return {
#                 "gradcam": encode_image_to_base64(cv2.cvtColor(grad_overlay, cv2.COLOR_BGR2RGB)),
#                 "ig": encode_image_to_base64(cv2.cvtColor(ig_overlay, cv2.COLOR_BGR2RGB)),
#                 "lime": encode_image_to_base64((lime_img * 255).astype(np.uint8))
#             }
#
#         densenet_output = process(mri_densenet, pred1, 'conv5_block32_concat')
#         resnet_output = process(mri_resnet, pred2, 'conv5_block3_out')
#
#         return jsonify({
#             "gradcam": {
#                 "densenet": densenet_output["gradcam"],
#                 "resnet": resnet_output["gradcam"]
#             },
#             "ig": {
#                 "densenet": densenet_output["ig"],
#                 "resnet": resnet_output["ig"]
#             },
#             "lime": {
#                 "densenet": densenet_output["lime"],
#                 "resnet": resnet_output["lime"]
#             }
#         })
#
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
#
#
# @app.route('/predict_pet', methods=['POST'])
# def predict_pet():
#     try:
#         file = request.files['file']
#         path = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
#         file.save(path)
#
#         img_input, _ = preprocess_image(path)
#         pred = pet_model.predict(img_input)[0]
#         pred_idx = np.argmax(pred)
#
#         app.config['PET_PATH'] = path
#         return jsonify({
#             "predicted_label": CLASSES[pred_idx],
#             "confidence": round(float(pred[pred_idx]) * 100, 2),
#             "actual_label": "Unknown",
#             "all_confidences": {CLASSES[i]: round(float(pred[i] * 100), 2) for i in range(len(CLASSES))}
#         })
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
#
# @app.route('/explain_pet', methods=['GET'])
# def explain_pet():
#     try:
#         path = app.config['PET_PATH']
#         img_input, img_raw = preprocess_image(path)
#         pred_class = np.argmax(pet_model.predict(img_input)[0])
#
#         grad = grad_cam(img_raw, pet_model, pred_class, 'block5_conv4')
#         grad_overlay = apply_matplotlib_colormap(np.clip(grad * get_brain_mask(img_raw), 0, 1))
#         grad_overlay = cv2.addWeighted((img_raw * 255).astype(np.uint8), 0.6, grad_overlay, 0.4, 0)
#
#         ig = integrated_gradients(img_raw, pet_model, pred_class)
#         ig_overlay = apply_matplotlib_colormap(ig)
#         ig_overlay = cv2.addWeighted((img_raw * 255).astype(np.uint8), 0.7, ig_overlay, 0.3, 0)
#
#         lime_img = lime_explainer(img_raw, pet_model)
#
#         return jsonify({
#             "gradcam": encode_image_to_base64(cv2.cvtColor(grad_overlay, cv2.COLOR_BGR2RGB)),
#             "ig": encode_image_to_base64(cv2.cvtColor(ig_overlay, cv2.COLOR_BGR2RGB)),
#             "lime": encode_image_to_base64((lime_img * 255).astype(np.uint8))
#         })
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
#
#
# @app.route('/predict_both', methods=['POST'])
# def predict_both():
#     try:
#         mri_file = request.files['mri']
#         pet_file = request.files['pet']
#         mri_path = os.path.join(UPLOAD_FOLDER, secure_filename("mri_" + mri_file.filename))
#         pet_path = os.path.join(UPLOAD_FOLDER, secure_filename("pet_" + pet_file.filename))
#         mri_file.save(mri_path)
#         pet_file.save(pet_path)
#
#         mri_input, _ = preprocess_image(mri_path)
#         pet_input, _ = preprocess_image(pet_path)
#
#         # MRI ensemble
#         mri_pred1 = mri_densenet.predict(mri_input)[0]
#         mri_pred2 = mri_resnet.predict(mri_input)[0]
#         mri_ensemble = (mri_pred1 + mri_pred2) / 2
#
#         # PET prediction
#         pet_pred = pet_model.predict(pet_input)[0]
#
#         # Decision-level fusion
#         final_probs = (mri_ensemble + pet_pred) / 2
#         pred_idx = np.argmax(final_probs)
#
#         # Store paths for explanation
#         app.config['MRI_PATH_BOTH'] = mri_path
#         app.config['PET_PATH_BOTH'] = pet_path
#
#         return jsonify({
#             "fused": {
#                 "predicted_label": CLASSES[pred_idx],
#                 "confidence": round(float(final_probs[pred_idx]) * 100, 2),
#                 "all_confidences": {CLASSES[i]: round(float(final_probs[i] * 100), 2) for i in range(len(CLASSES))}
#             },
#             "mri": {
#                 "predicted_label": CLASSES[np.argmax(mri_ensemble)],
#                 "confidence": round(float(np.max(mri_ensemble)) * 100, 2),
#                 "all_confidences": {CLASSES[i]: round(float(mri_ensemble[i] * 100), 2) for i in range(len(CLASSES))}
#             },
#             "pet": {
#                 "predicted_label": CLASSES[np.argmax(pet_pred)],
#                 "confidence": round(float(np.max(pet_pred)) * 100, 2),
#                 "all_confidences": {CLASSES[i]: round(float(pet_pred[i] * 100), 2) for i in range(len(CLASSES))}
#             }
#         })
#
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
#
# @app.route('/explain_both', methods=['GET'])
# def explain_both():
#     try:
#         mri_path = app.config['MRI_PATH_BOTH']
#         pet_path = app.config['PET_PATH_BOTH']
#
#         mri_input, mri_raw = preprocess_image(mri_path)
#         pet_input, pet_raw = preprocess_image(pet_path)
#
#         mri_pred1 = np.argmax(mri_densenet.predict(mri_input)[0])
#         mri_pred2 = np.argmax(mri_resnet.predict(mri_input)[0])
#         pet_pred = np.argmax(pet_model.predict(pet_input)[0])
#
#         def process(image, model, pred_class, layer_name=None):
#             grad = grad_cam(image, model, pred_class, layer_name) if layer_name else np.zeros((IMG_SIZE, IMG_SIZE))
#             grad_overlay = apply_matplotlib_colormap(np.clip(grad * get_brain_mask(image), 0, 1))
#             grad_overlay = cv2.addWeighted((image * 255).astype(np.uint8), 0.6, grad_overlay, 0.4, 0)
#
#             ig = integrated_gradients(image, model, pred_class)
#             ig = ig ** 0.5
#             ig_overlay = apply_matplotlib_colormap(ig)
#             ig_overlay = cv2.addWeighted((image * 255).astype(np.uint8), 0.4, ig_overlay, 0.6, 0)
#
#             lime_img = lime_explainer(image, model)
#
#             return {
#                 "gradcam": encode_image_to_base64(cv2.cvtColor(grad_overlay, cv2.COLOR_BGR2RGB)),
#                 "ig": encode_image_to_base64(cv2.cvtColor(ig_overlay, cv2.COLOR_BGR2RGB)),
#                 "lime": encode_image_to_base64((lime_img * 255).astype(np.uint8))
#             }
#
#         mri_dense_out = process(mri_raw, mri_densenet, mri_pred1, 'conv5_block32_concat')
#         mri_resnet_out = process(mri_raw, mri_resnet, mri_pred2, 'conv5_block3_out')
#         pet_out = process(pet_raw, pet_model, pet_pred, 'block5_conv4')
#
#         return jsonify({
#             "mri": {
#                 "densenet": mri_dense_out,
#                 "resnet": mri_resnet_out
#             },
#             "pet": pet_out
#         })
#
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
#
#
# if __name__ == '__main__':
#     app.run(host="0.0.0.0", port=5000, debug=True)   #correct version

# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import os
# import numpy as np
# import tensorflow as tf
# import cv2
# from werkzeug.utils import secure_filename
# from lime import lime_image
# from skimage.segmentation import mark_boundaries
# import base64
# import matplotlib.cm as cm
# from concurrent.futures import ThreadPoolExecutor
#
#
# app = Flask(__name__)
# CORS(app)
#
# # Paths
# MRI_MODEL_PATH_DENSENET = "D:/SAI/MRI_DenseNet201_CBAM_best.h5"
# MRI_MODEL_PATH_RESNET = "D:/SAI/MRI_ResNet50_CBAM_E.h5"
# PET_MODEL_PATH = "D:/SAI/PET_VGG19_CBAM_Enhanced.h5"
# UPLOAD_FOLDER = "uploads"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
#
# # Constants
# IMG_SIZE = 128
# CLASSES = ['AD', 'CN', 'EMCI', 'LMCI']
#
# # Load Models
# mri_densenet = tf.keras.models.load_model(MRI_MODEL_PATH_DENSENET)
# mri_resnet = tf.keras.models.load_model(MRI_MODEL_PATH_RESNET)
# pet_model = tf.keras.models.load_model(PET_MODEL_PATH)
#
# # Helpers
# def encode_image_to_base64(image):
#     if image.dtype != np.uint8:
#         image = (image * 255).astype(np.uint8)
#     _, buffer = cv2.imencode('.png', image)
#     return base64.b64encode(buffer).decode('utf-8')
#
# def apply_matplotlib_colormap(heatmap):
#     cmap = cm.get_cmap('jet')
#     colored_heatmap = cmap(heatmap)[:, :, :3]
#     return (colored_heatmap * 255).astype(np.uint8)
#
# def preprocess_image(image_path):
#     img = cv2.imread(image_path, cv2.IMREAD_COLOR)
#     if img is None:
#         raise ValueError("Cannot load image")
#     img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
#     img = img.astype(np.float32) / 255.0
#     return np.expand_dims(img, axis=0), img
#
# def get_brain_mask(image, threshold=0.1):
#     return (np.mean(image, axis=-1) > threshold).astype(np.float32)
# #
# # def integrated_gradients(image, model, class_idx):
# #     image_tensor = tf.convert_to_tensor(image[np.newaxis, ...], dtype=tf.float32)
# #     with tf.GradientTape() as tape:
# #         tape.watch(image_tensor)
# #         preds = model(image_tensor)
# #         target = preds[:, class_idx]
# #     grads = tape.gradient(target, image_tensor)
# #     attribution = np.mean(np.abs(grads.numpy().squeeze()), axis=-1)
# #     attribution = np.nan_to_num(attribution)
# #     attribution = np.clip(attribution, 0, 1)
# #     if np.max(attribution) != 0:
# #         attribution = (attribution - np.min(attribution)) / (np.max(attribution) - np.min(attribution) + 1e-8)
# #     return attribution * get_brain_mask(image)
#
# def integrated_gradients(image, model, class_index, steps=50):
#     image = image.astype(np.float32)
#     baseline = np.zeros_like(image).astype(np.float32)  # Black image baseline
#
#     # Scale inputs between baseline and input image
#     interpolated_images = [baseline + (float(i) / steps) * (image - baseline) for i in range(steps + 1)]
#     interpolated_images = tf.convert_to_tensor(np.array(interpolated_images), dtype=tf.float32)
#
#     with tf.GradientTape() as tape:
#         tape.watch(interpolated_images)
#         preds = model(interpolated_images)
#         target = preds[:, class_index]
#
#     grads = tape.gradient(target, interpolated_images).numpy()
#     avg_grads = np.mean(grads, axis=0)  # Average across steps
#     integrated_grads = (image - baseline) * avg_grads
#
#     # Aggregate across channels
#     attributions = np.mean(np.abs(integrated_grads), axis=-1)
#     attributions = np.clip(attributions, 0, 1)
#
#     if np.max(attributions) > 0:
#         attributions = (attributions - np.min(attributions)) / (np.max(attributions) - np.min(attributions) + 1e-8)
#
#     # Mask out background
#     return attributions * get_brain_mask(image)
#
#
# def grad_cam(image, model, class_idx, layer_name):
#     image_tensor = tf.convert_to_tensor(image[np.newaxis, ...], dtype=tf.float32)
#     grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])
#     with tf.GradientTape() as tape:
#         conv_outputs, predictions = grad_model(image_tensor)
#         loss = predictions[:, class_idx]
#     grads = tape.gradient(loss, conv_outputs)
#     pooled_grads = tf.reduce_mean(grads, axis=(1, 2))
#     heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1).numpy().squeeze()
#     heatmap = np.maximum(heatmap, 0)
#     if np.max(heatmap) != 0:
#         heatmap /= np.max(heatmap)
#     return cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
#
# # def lime_explainer(image, model):
# #     explainer = lime_image.LimeImageExplainer()
# #     explanation = explainer.explain_instance(
# #         image.astype('double'),
# #         model.predict,
# #         top_labels=len(CLASSES),
# #         hide_color=0,
# #         num_samples=1000
# #     )
# #     pred_class = np.argmax(model.predict(image[np.newaxis, ...]))
# #     temp, mask = explanation.get_image_and_mask(pred_class, positive_only=True, num_features=5, hide_rest=False)
# #     return mark_boundaries(temp, mask, color=(1, 1, 0))
#
# def lime_explainer(image, model):
#     explainer = lime_image.LimeImageExplainer()
#     explanation = explainer.explain_instance(
#         image.astype('double'),
#         model.predict,
#         top_labels=len(CLASSES),
#         hide_color=0,        # You can also try None
#         num_samples=500,
#         batch_size=32
#
#     )
#
#     pred_class = np.argmax(model.predict(image[np.newaxis, ...]))
#     temp, mask = explanation.get_image_and_mask(
#         pred_class,
#         positive_only=True,
#         num_features=8,       # Try increasing number of features
#         hide_rest=False
#     )
#
#     # Make temp uint8 for better visualization
#     temp_uint8 = (temp * 255).astype(np.uint8)
#     boundaries = mark_boundaries(temp_uint8, mask, color=(1, 1, 0), mode='thick')
#
#     return boundaries
#
#
# # --- Routes ---
#
# @app.route('/')
# def home():
#     return "XAI MRI & PET API for Alzheimer's detection is live!"
#
# @app.route('/predict_mri', methods=['POST'])
# def predict_mri():
#     try:
#         file = request.files['file']
#         path = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
#         file.save(path)
#
#         img_input, _ = preprocess_image(path)
#
#         pred1 = mri_densenet.predict(img_input)[0]
#         pred2 = mri_resnet.predict(img_input)[0]
#         ensemble = (pred1 + pred2) / 2
#         pred_idx = np.argmax(ensemble)
#
#         app.config['MRI_PATH'] = path
#         return jsonify({
#             "predicted_label": CLASSES[pred_idx],
#             "confidence": round(float(ensemble[pred_idx]) * 100, 2),
#             "actual_label": "Unknown",
#             "all_confidences": {CLASSES[i]: round(float(ensemble[i] * 100), 2) for i in range(len(CLASSES))}
#         })
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
#
#
# # @app.route('/explain_mri', methods=['GET'])
# # def explain_mri():
# #     try:
# #         path = app.config['MRI_PATH']
# #         img_input, img_raw = preprocess_image(path)
# #
# #         pred1 = np.argmax(mri_densenet.predict(img_input)[0])
# #         pred2 = np.argmax(mri_resnet.predict(img_input)[0])
# #
# #         def process(model, pred_class, layer_name):
# #             grad = grad_cam(img_raw, model, pred_class, layer_name)
# #             grad_overlay = apply_matplotlib_colormap(np.clip(grad * get_brain_mask(img_raw), 0, 1))
# #             grad_overlay = cv2.addWeighted((img_raw * 255).astype(np.uint8), 0.6, grad_overlay, 0.4, 0)
# #
# #             # ig = integrated_gradients(img_raw, model, pred_class)
# #             # ig_overlay = apply_matplotlib_colormap(ig)
# #             # ig_overlay = cv2.addWeighted((img_raw * 255).astype(np.uint8), 0.7, ig_overlay, 0.3, 0)
# #
# #             ig = integrated_gradients(img_raw, model, pred_class)
# #
# #             # Apply gamma correction to boost weak signals
# #             ig = ig ** 0.5  # You can also try 0.4 or 0.6 depending on visibility
# #
# #             ig_overlay = apply_matplotlib_colormap(ig)
# #
# #             # Stronger overlay (more heatmap visibility)
# #             ig_overlay = cv2.addWeighted((img_raw * 255).astype(np.uint8), 0.4, ig_overlay, 0.6, 0)
# #
# #             lime_img = lime_explainer(img_raw, model)
# #
# #             return {
# #                 "gradcam": encode_image_to_base64(cv2.cvtColor(grad_overlay, cv2.COLOR_BGR2RGB)),
# #                 "ig": encode_image_to_base64(cv2.cvtColor(ig_overlay, cv2.COLOR_BGR2RGB)),
# #                 "lime": encode_image_to_base64((lime_img * 255).astype(np.uint8))
# #             }
# #
# #         densenet_output = process(mri_densenet, pred1, 'conv5_block32_concat')
# #         resnet_output = process(mri_resnet, pred2, 'conv5_block3_out')
# #
# #         return jsonify({
# #             "gradcam": {
# #                 "densenet": densenet_output["gradcam"],
# #                 "resnet": resnet_output["gradcam"]
# #             },
# #             "ig": {
# #                 "densenet": densenet_output["ig"],
# #                 "resnet": resnet_output["ig"]
# #             },
# #             "lime": {
# #                 "densenet": densenet_output["lime"],
# #                 "resnet": resnet_output["lime"]
# #             }
# #         })
# #
# #     except Exception as e:
# #         return jsonify({"error": str(e)}), 500
#
# @app.route('/explain_mri', methods=['GET'])
# def explain_mri():
#     try:
#         path = app.config['MRI_PATH']
#         img_input, img_raw = preprocess_image(path)
#
#         pred1 = np.argmax(mri_densenet.predict(img_input)[0])
#         pred2 = np.argmax(mri_resnet.predict(img_input)[0])
#
#         def process(model, pred_class, layer_name):
#             with ThreadPoolExecutor() as executor:
#                 gradcam_future = executor.submit(grad_cam, img_raw, model, pred_class, layer_name)
#                 ig_future = executor.submit(integrated_gradients, img_raw, model, pred_class)
#                 lime_future = executor.submit(lime_explainer, img_raw, model)
#
#                 # Grad-CAM
#                 grad = gradcam_future.result()
#                 grad_overlay = apply_matplotlib_colormap(np.clip(grad * get_brain_mask(img_raw), 0, 1))
#                 grad_overlay = cv2.addWeighted((img_raw * 255).astype(np.uint8), 0.6, grad_overlay, 0.4, 0)
#                 gradcam_encoded = encode_image_to_base64(cv2.cvtColor(grad_overlay, cv2.COLOR_BGR2RGB))
#
#                 # Integrated Gradients
#                 ig = ig_future.result() ** 0.5  # Gamma correction
#                 ig_overlay = apply_matplotlib_colormap(ig)
#                 ig_overlay = cv2.addWeighted((img_raw * 255).astype(np.uint8), 0.4, ig_overlay, 0.6, 0)
#                 ig_encoded = encode_image_to_base64(cv2.cvtColor(ig_overlay, cv2.COLOR_BGR2RGB))
#
#                 # LIME
#                 lime_img = lime_future.result()
#                 lime_encoded = encode_image_to_base64((lime_img * 255).astype(np.uint8))
#
#             return {
#                 "gradcam": gradcam_encoded,
#                 "ig": ig_encoded,
#                 "lime": lime_encoded
#             }
#
#         densenet_output = process(mri_densenet, pred1, 'conv5_block32_concat')
#         resnet_output = process(mri_resnet, pred2, 'conv5_block3_out')
#
#         return jsonify({
#             "gradcam": {
#                 "densenet": densenet_output["gradcam"],
#                 "resnet": resnet_output["gradcam"]
#             },
#             "ig": {
#                 "densenet": densenet_output["ig"],
#                 "resnet": resnet_output["ig"]
#             },
#             "lime": {
#                 "densenet": densenet_output["lime"],
#                 "resnet": resnet_output["lime"]
#             }
#         })
#
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
#
#
#
# @app.route('/predict_pet', methods=['POST'])
# def predict_pet():
#     try:
#         file = request.files['file']
#         path = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
#         file.save(path)
#
#         img_input, _ = preprocess_image(path)
#         pred = pet_model.predict(img_input)[0]
#         pred_idx = np.argmax(pred)
#
#         app.config['PET_PATH'] = path
#         return jsonify({
#             "predicted_label": CLASSES[pred_idx],
#             "confidence": round(float(pred[pred_idx]) * 100, 2),
#             "actual_label": "Unknown",
#             "all_confidences": {CLASSES[i]: round(float(pred[i] * 100), 2) for i in range(len(CLASSES))}
#         })
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
#
# @app.route('/explain_pet', methods=['GET'])
# def explain_pet():
#     try:
#         path = app.config['PET_PATH']
#         img_input, img_raw = preprocess_image(path)
#         pred_class = np.argmax(pet_model.predict(img_input)[0])
#
#         grad = grad_cam(img_raw, pet_model, pred_class, 'block5_conv4')
#         grad_overlay = apply_matplotlib_colormap(np.clip(grad * get_brain_mask(img_raw), 0, 1))
#         grad_overlay = cv2.addWeighted((img_raw * 255).astype(np.uint8), 0.6, grad_overlay, 0.4, 0)
#
#         ig = integrated_gradients(img_raw, pet_model, pred_class)
#         ig_overlay = apply_matplotlib_colormap(ig)
#         ig_overlay = cv2.addWeighted((img_raw * 255).astype(np.uint8), 0.7, ig_overlay, 0.3, 0)
#
#         lime_img = lime_explainer(img_raw, pet_model)
#
#         return jsonify({
#             "gradcam": encode_image_to_base64(cv2.cvtColor(grad_overlay, cv2.COLOR_BGR2RGB)),
#             "ig": encode_image_to_base64(cv2.cvtColor(ig_overlay, cv2.COLOR_BGR2RGB)),
#             "lime": encode_image_to_base64((lime_img * 255).astype(np.uint8))
#         })
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
#
#
# @app.route('/predict_both', methods=['POST'])
# def predict_both():
#     try:
#         mri_file = request.files['mri']
#         pet_file = request.files['pet']
#         mri_path = os.path.join(UPLOAD_FOLDER, secure_filename("mri_" + mri_file.filename))
#         pet_path = os.path.join(UPLOAD_FOLDER, secure_filename("pet_" + pet_file.filename))
#         mri_file.save(mri_path)
#         pet_file.save(pet_path)
#
#         mri_input, _ = preprocess_image(mri_path)
#         pet_input, _ = preprocess_image(pet_path)
#
#         # MRI ensemble
#         mri_pred1 = mri_densenet.predict(mri_input)[0]
#         mri_pred2 = mri_resnet.predict(mri_input)[0]
#         mri_ensemble = (mri_pred1 + mri_pred2) / 2
#
#         # PET prediction
#         pet_pred = pet_model.predict(pet_input)[0]
#
#         # Decision-level fusion
#         final_probs = (mri_ensemble + pet_pred) / 2
#         pred_idx = np.argmax(final_probs)
#
#         # Store paths for explanation
#         app.config['MRI_PATH_BOTH'] = mri_path
#         app.config['PET_PATH_BOTH'] = pet_path
#
#         return jsonify({
#             "fused": {
#                 "predicted_label": CLASSES[pred_idx],
#                 "confidence": round(float(final_probs[pred_idx]) * 100, 2),
#                 "all_confidences": {CLASSES[i]: round(float(final_probs[i] * 100), 2) for i in range(len(CLASSES))}
#             },
#             "mri": {
#                 "predicted_label": CLASSES[np.argmax(mri_ensemble)],
#                 "confidence": round(float(np.max(mri_ensemble)) * 100, 2),
#                 "all_confidences": {CLASSES[i]: round(float(mri_ensemble[i] * 100), 2) for i in range(len(CLASSES))}
#             },
#             "pet": {
#                 "predicted_label": CLASSES[np.argmax(pet_pred)],
#                 "confidence": round(float(np.max(pet_pred)) * 100, 2),
#                 "all_confidences": {CLASSES[i]: round(float(pet_pred[i] * 100), 2) for i in range(len(CLASSES))}
#             }
#         })
#
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
#
# # @app.route('/explain_both', methods=['GET'])
# # def explain_both():
# #     try:
# #         mri_path = app.config['MRI_PATH_BOTH']
# #         pet_path = app.config['PET_PATH_BOTH']
# #
# #         mri_input, mri_raw = preprocess_image(mri_path)
# #         pet_input, pet_raw = preprocess_image(pet_path)
# #
# #         mri_pred1 = np.argmax(mri_densenet.predict(mri_input)[0])
# #         mri_pred2 = np.argmax(mri_resnet.predict(mri_input)[0])
# #         pet_pred = np.argmax(pet_model.predict(pet_input)[0])
# #
# #         def process(image, model, pred_class, layer_name=None):
# #             grad = grad_cam(image, model, pred_class, layer_name) if layer_name else np.zeros((IMG_SIZE, IMG_SIZE))
# #             grad_overlay = apply_matplotlib_colormap(np.clip(grad * get_brain_mask(image), 0, 1))
# #             grad_overlay = cv2.addWeighted((image * 255).astype(np.uint8), 0.6, grad_overlay, 0.4, 0)
# #
# #             ig = integrated_gradients(image, model, pred_class)
# #             ig = ig ** 0.5
# #             ig_overlay = apply_matplotlib_colormap(ig)
# #             ig_overlay = cv2.addWeighted((image * 255).astype(np.uint8), 0.4, ig_overlay, 0.6, 0)
# #
# #             lime_img = lime_explainer(image, model)
# #
# #             return {
# #                 "gradcam": encode_image_to_base64(cv2.cvtColor(grad_overlay, cv2.COLOR_BGR2RGB)),
# #                 "ig": encode_image_to_base64(cv2.cvtColor(ig_overlay, cv2.COLOR_BGR2RGB)),
# #                 "lime": encode_image_to_base64((lime_img * 255).astype(np.uint8))
# #             }
# #
# #         mri_dense_out = process(mri_raw, mri_densenet, mri_pred1, 'conv5_block32_concat')
# #         mri_resnet_out = process(mri_raw, mri_resnet, mri_pred2, 'conv5_block3_out')
# #         pet_out = process(pet_raw, pet_model, pet_pred, 'block5_conv4')
# #
# #         return jsonify({
# #             "mri": {
# #                 "densenet": mri_dense_out,
# #                 "resnet": mri_resnet_out
# #             },
# #             "pet": pet_out
# #         })
# #
# #     except Exception as e:
# #         return jsonify({"error": str(e)}), 500
#
# from concurrent.futures import ThreadPoolExecutor
#
# @app.route('/explain_both', methods=['GET'])
# def explain_both():
#     try:
#         mri_path = app.config['MRI_PATH_BOTH']
#         pet_path = app.config['PET_PATH_BOTH']
#
#         mri_input, mri_raw = preprocess_image(mri_path)
#         pet_input, pet_raw = preprocess_image(pet_path)
#
#         mri_pred1 = np.argmax(mri_densenet.predict(mri_input)[0])
#         mri_pred2 = np.argmax(mri_resnet.predict(mri_input)[0])
#         pet_pred = np.argmax(pet_model.predict(pet_input)[0])
#
#         def process(image, model, pred_class, layer_name=None):
#             with ThreadPoolExecutor() as executor:
#                 gradcam_future = executor.submit(
#                     grad_cam, image, model, pred_class, layer_name) if layer_name else None
#                 ig_future = executor.submit(integrated_gradients, image, model, pred_class)
#                 lime_future = executor.submit(lime_explainer, image, model)
#
#                 # Grad-CAM processing
#                 if gradcam_future:
#                     grad = gradcam_future.result()
#                     grad_overlay = apply_matplotlib_colormap(np.clip(grad * get_brain_mask(image), 0, 1))
#                     grad_overlay = cv2.addWeighted((image * 255).astype(np.uint8), 0.6, grad_overlay, 0.4, 0)
#                     gradcam_encoded = encode_image_to_base64(cv2.cvtColor(grad_overlay, cv2.COLOR_BGR2RGB))
#                 else:
#                     gradcam_encoded = None
#
#                 # IG processing
#                 ig = ig_future.result() ** 0.5
#                 ig_overlay = apply_matplotlib_colormap(ig)
#                 ig_overlay = cv2.addWeighted((image * 255).astype(np.uint8), 0.4, ig_overlay, 0.6, 0)
#                 ig_encoded = encode_image_to_base64(cv2.cvtColor(ig_overlay, cv2.COLOR_BGR2RGB))
#
#                 # LIME processing
#                 lime_img = lime_future.result()
#                 lime_encoded = encode_image_to_base64((lime_img * 255).astype(np.uint8))
#
#             return {
#                 "gradcam": gradcam_encoded,
#                 "ig": ig_encoded,
#                 "lime": lime_encoded
#             }
#
#         # Process explanations in parallel for each model
#         mri_dense_out = process(mri_raw, mri_densenet, mri_pred1, 'conv5_block32_concat')
#         mri_resnet_out = process(mri_raw, mri_resnet, mri_pred2, 'conv5_block3_out')
#         pet_out = process(pet_raw, pet_model, pet_pred, 'block5_conv4')
#
#         return jsonify({
#             "mri": {
#                 "densenet": mri_dense_out,
#                 "resnet": mri_resnet_out
#             },
#             "pet": pet_out
#         })
#
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
#
#
#
# if __name__ == '__main__':
#     app.run(host="0.0.0.0", port=5000, debug=True) #Multi thrading and Batch

# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import os
# import numpy as np
# import tensorflow as tf
# import cv2
# from werkzeug.utils import secure_filename
# from lime import lime_image
# from skimage.segmentation import mark_boundaries
# import base64
# import matplotlib.cm as cm
# from concurrent.futures import ThreadPoolExecutor # Import ThreadPoolExecutor
#
# # --- (Keep Flask app setup, CORS, Paths, Constants, Model Loading, Helpers as before) ---
#
# app = Flask(__name__)
# CORS(app)
#
# # Paths
# MRI_MODEL_PATH_DENSENET = "D:/SAI/MRI_DenseNet201_CBAM_best.h5"
# MRI_MODEL_PATH_RESNET = "D:/SAI/MRI_ResNet50_CBAM_E.h5"
# PET_MODEL_PATH = "D:/SAI/PET_VGG19_CBAM_Enhanced.h5"
# UPLOAD_FOLDER = "uploads"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
#
# # Constants
# IMG_SIZE = 128
# CLASSES = ['AD', 'CN', 'EMCI', 'LMCI']
#
# # Load Models (Consider lazy loading if startup time is an issue, but not primary concern here)
# print("Loading models...")
# mri_densenet = tf.keras.models.load_model(MRI_MODEL_PATH_DENSENET)
# mri_resnet = tf.keras.models.load_model(MRI_MODEL_PATH_RESNET)
# pet_model = tf.keras.models.load_model(PET_MODEL_PATH)
# print("Models loaded.")
#
# # --- Helper Functions (encode_image_to_base64, apply_matplotlib_colormap, preprocess_image, get_brain_mask) ---
#
# def encode_image_to_base64(image):
#     # Ensure image is uint8 before encoding
#     if image.dtype != np.uint8:
#         # Handle potential float images from LIME or overlays
#         if image.max() <= 1.0 and image.min() >= 0.0:
#              image = (image * 255).astype(np.uint8)
#         else:
#             # Attempt safe conversion if values might be outside 0-1 or 0-255
#             image = np.clip(image, 0, 255).astype(np.uint8)
#
#     _, buffer = cv2.imencode('.png', image)
#     return base64.b64encode(buffer).decode('utf-8')
#
# def apply_matplotlib_colormap(heatmap):
#     cmap = cm.get_cmap('jet')
#     # Normalize heatmap to 0-1 range before applying colormap
#     heatmap_normalized = np.clip(heatmap, 0, 1)
#     colored_heatmap = cmap(heatmap_normalized)[:, :, :3] # Take only RGB
#     return (colored_heatmap * 255).astype(np.uint8)
#
# def preprocess_image(image_path):
#     img = cv2.imread(image_path, cv2.IMREAD_COLOR)
#     if img is None:
#         raise ValueError(f"Cannot load image: {image_path}")
#     # Ensure input is 3 channel BGR
#     if len(img.shape) == 2: # Grayscale
#         img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#     elif img.shape[2] == 4: # RGBA
#         img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
#
#     img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
#     img_normalized = img_resized.astype(np.float32) / 255.0
#     return np.expand_dims(img_normalized, axis=0), img_normalized # Return input tensor and raw-like (resized, normalized)
#
# def get_brain_mask(image, threshold=0.1):
#     if len(image.shape) == 3 and image.shape[2] == 3:
#          gray_image = np.mean(image, axis=-1)
#     elif len(image.shape) == 2:
#          gray_image = image
#     else:
#          # Fallback or raise error if unexpected shape
#          gray_image = image[:,:,0] # Assume first channel if unsure
#
#     mask = (gray_image > threshold).astype(np.float32)
#     return mask
#
# # --- Explanation Functions (with Tuned Parameters) ---
#
# def integrated_gradients(image, model, class_index, steps=50): # Reduced steps
#     image = image.astype(np.float32)
#     baseline = np.zeros_like(image).astype(np.float32)
#     interpolated_images = [baseline + (float(i) / steps) * (image - baseline) for i in range(steps + 1)]
#     interpolated_images = tf.convert_to_tensor(np.array(interpolated_images), dtype=tf.float32)
#
#     with tf.GradientTape() as tape:
#         tape.watch(interpolated_images)
#         preds = model(interpolated_images)
#         target = preds[:, class_index]
#
#     grads = tape.gradient(target, interpolated_images)
#     if grads is None:
#          print(f"Warning: IG Gradients are None for class {class_index}. Returning zeros.")
#          return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
#
#     grads = grads.numpy()
#     avg_grads = np.mean(grads, axis=0)
#     integrated_grads = (image - baseline) * avg_grads
#     attributions = np.mean(np.abs(integrated_grads), axis=-1)
#     attributions = np.nan_to_num(attributions) # Handle potential NaNs
#
#     # Normalize 0-1
#     min_val, max_val = np.min(attributions), np.max(attributions)
#     if max_val > min_val:
#         attributions = (attributions - min_val) / (max_val - min_val + 1e-8)
#     else:
#         attributions = np.zeros_like(attributions) # Avoid division by zero if flat
#
#     # Mask out background AFTER normalization
#     brain_mask = get_brain_mask(image)
#     return attributions * brain_mask
#
#
# def grad_cam(image, model, class_idx, layer_name):
#     image_tensor = tf.convert_to_tensor(image[np.newaxis, ...], dtype=tf.float32)
#
#     # Find the layer if name is partial or needs checking
#     try:
#         target_layer = model.get_layer(layer_name)
#     except ValueError:
#         print(f"Warning: Layer '{layer_name}' not found directly. Searching...")
#         found = False
#         for layer in reversed(model.layers): # Search backwards (likely deeper)
#              if layer_name in layer.name and isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Add, tf.keras.layers.Concatenate)): # Common layer types
#                   target_layer = layer
#                   print(f"Using layer: {target_layer.name}")
#                   found = True
#                   break
#         if not found:
#              print(f"Error: Could not find suitable layer containing '{layer_name}'. Using last conv layer.")
#              # Fallback: try to find the last convolutional layer
#              for layer in reversed(model.layers):
#                   if isinstance(layer, tf.keras.layers.Conv2D):
#                        target_layer = layer
#                        print(f"Fallback: Using last Conv2D layer: {target_layer.name}")
#                        break
#              if target_layer is None: # Should not happen if model has Conv layers
#                  raise ValueError("Cannot find any Conv layer for GradCAM")
#
#     grad_model = tf.keras.models.Model([model.inputs], [target_layer.output, model.output])
#
#     with tf.GradientTape() as tape:
#         conv_outputs, predictions = grad_model(image_tensor)
#         if class_idx >= predictions.shape[1]:
#              print(f"Warning: class_idx {class_idx} out of bounds for prediction shape {predictions.shape}. Using index 0.")
#              class_idx = 0
#         loss = predictions[:, class_idx]
#
#     grads = tape.gradient(loss, conv_outputs)
#     if grads is None:
#          print(f"Warning: GradCAM Gradients are None for layer {target_layer.name}, class {class_idx}. Returning zeros.")
#          return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
#
#     # Handle potential batch dimension in grads if input was batched differently
#     if len(grads.shape) == 4: # (batch, h, w, c) - expected
#         pooled_grads = tf.reduce_mean(grads, axis=(1, 2)) # Pool over spatial dims H, W
#     elif len(grads.shape) == 3: # Might happen in some architectures (h, w, c)
#         pooled_grads = tf.reduce_mean(grads, axis=(0, 1)) # Pool over spatial dims H, W
#         pooled_grads = tf.expand_dims(pooled_grads, axis=0) # Add back batch dim
#     else:
#         print(f"Warning: Unexpected grads shape in GradCAM: {grads.shape}. Returning zeros.")
#         return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
#
#     # Ensure conv_outputs also has batch dim
#     if len(conv_outputs.shape) == 3:
#          conv_outputs = tf.expand_dims(conv_outputs, axis=0)
#
#     # Element-wise multiply pooled grads with feature map channels
#     heatmap = tf.multiply(pooled_grads[..., tf.newaxis, tf.newaxis, :], conv_outputs)
#     heatmap = tf.reduce_sum(heatmap, axis=-1).numpy().squeeze() # Sum over channels
#     heatmap = np.maximum(heatmap, 0) # ReLU
#
#     # Normalize 0-1
#     max_val = np.max(heatmap)
#     if max_val > 0:
#         heatmap /= max_val
#
#     return cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE)) # Resize to original image size
#
#
# def lime_explainer(image, model):
#     explainer = lime_image.LimeImageExplainer()
#     # Use a lambda to ensure predict function has correct output shape for LIME
#     def predict_fn_for_lime(images):
#         return model.predict(images)
#
#     explanation = explainer.explain_instance(
#         image.astype('double'),
#         predict_fn_for_lime, # Use the wrapper
#         top_labels=len(CLASSES),
#         hide_color=0,
#         num_samples=500,  # <<< REDUCED SAMPLES
#         random_seed=42 # Add for reproducibility
#     )
#
#     # Predict once outside LIME to get the top class reliably
#     preds = model.predict(image[np.newaxis, ...])
#     pred_class = np.argmax(preds[0])
#
#     temp, mask = explanation.get_image_and_mask(
#         pred_class,
#         positive_only=True,
#         num_features=5, # <<< REDUCED FEATURES
#         hide_rest=False,
#         min_weight=0.05 # Add a minimum weight threshold
#     )
#
#     temp_uint8 = (temp * 255).astype(np.uint8)
#     # Ensure mask is boolean or 0/1
#     mask_bool = mask > 0
#     # Use mode='inner' or 'outer' for thinner/thicker boundaries if needed
#     boundaries = mark_boundaries(temp_uint8, mask_bool, color=(0, 1, 0), mode='thick') # Green boundaries (BGR)
#
#     return boundaries # Returns float 0-1 range image
#
# # --- Processing Function (called by explain routes) ---
# def generate_explanations(image_raw, model, pred_class, layer_name=None):
#     """Generates GradCAM, IG, and LIME explanations for a single model."""
#     results = {}
#
#     # Grad-CAM
#     if layer_name:
#         try:
#             grad = grad_cam(image_raw, model, pred_class, layer_name)
#             # Apply brain mask *before* overlaying
#             grad_masked = np.clip(grad * get_brain_mask(image_raw), 0, 1)
#             grad_heatmap = apply_matplotlib_colormap(grad_masked)
#             grad_overlay = cv2.addWeighted((image_raw * 255).astype(np.uint8), 0.6, grad_heatmap, 0.4, 0)
#             results["gradcam"] = encode_image_to_base64(cv2.cvtColor(grad_overlay, cv2.COLOR_BGR2RGB))
#         except Exception as e:
#             print(f"Error generating Grad-CAM for {model.name}: {e}")
#             results["gradcam"] = None # Or a placeholder image
#     else:
#         results["gradcam"] = None # No layer name provided
#
#     # Integrated Gradients
#     try:
#         ig = integrated_gradients(image_raw, model, pred_class, steps=25) # Use tuned steps
#         # Optional: Apply gamma correction if needed (can sometimes help visibility)
#         # ig = ig ** 0.6
#         ig_heatmap = apply_matplotlib_colormap(ig) # IG already includes brain mask
#         ig_overlay = cv2.addWeighted((image_raw * 255).astype(np.uint8), 0.5, ig_heatmap, 0.5, 0) # Adjusted weights
#         results["ig"] = encode_image_to_base64(cv2.cvtColor(ig_overlay, cv2.COLOR_BGR2RGB))
#     except Exception as e:
#         print(f"Error generating Integrated Gradients for {model.name}: {e}")
#         results["ig"] = None
#
#     # LIME
#     try:
#         # LIME returns image with boundaries, already float 0-1
#         lime_img = lime_explainer(image_raw, model)
#         # Convert to RGB before encoding
#         lime_img_rgb = cv2.cvtColor((lime_img * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
#         results["lime"] = encode_image_to_base64(lime_img_rgb)
#     except Exception as e:
#         print(f"Error generating LIME for {model.name}: {e}")
#         results["lime"] = None
#
#     return results
#
#
# # --- Routes ---
#
# @app.route('/')
# def home():
#     return "XAI MRI & PET API for Alzheimer's detection is live!"
#
# # --- (predict_mri remains the same) ---
# @app.route('/predict_mri', methods=['POST'])
# def predict_mri():
#     if 'file' not in request.files:
#         return jsonify({"error": "No file part"}), 400
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({"error": "No selected file"}), 400
#
#     try:
#         filename = secure_filename(file.filename)
#         path = os.path.join(UPLOAD_FOLDER, filename)
#         file.save(path)
#
#         img_input, _ = preprocess_image(path) # Only need input tensor here
#
#         pred1 = mri_densenet.predict(img_input)[0]
#         pred2 = mri_resnet.predict(img_input)[0]
#         ensemble = (pred1 + pred2) / 2
#         pred_idx = np.argmax(ensemble)
#         pred_label = CLASSES[pred_idx]
#         confidence = round(float(ensemble[pred_idx]) * 100, 2)
#         all_confidences = {CLASSES[i]: round(float(ensemble[i] * 100), 2) for i in range(len(CLASSES))}
#
#         # Store path associated with this prediction *temporarily*.
#         # WARNING: This is NOT thread-safe if multiple users hit the API.
#         # A better approach would be to return the filename and have the explain
#         # endpoint accept the filename, or use a session/cache.
#         # For simplicity here, we keep app.config, but be aware of its limitations.
#         app.config['LAST_MRI_PATH'] = path
#         app.config['LAST_MRI_PRED_IDX'] = pred_idx # Store index too
#
#         return jsonify({
#             "predicted_label": pred_label,
#             "confidence": confidence,
#             "actual_label": "Unknown", # Assuming no ground truth provided
#             "all_confidences": all_confidences
#         })
#     except ValueError as ve:
#          return jsonify({"error": f"Image processing error: {str(ve)}"}), 400
#     except Exception as e:
#         import traceback
#         print(traceback.format_exc())
#         return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
#
#
# # --- UPDATED /explain_mri with Parallelism ---
# @app.route('/explain_mri', methods=['GET'])
# def explain_mri():
#     # Retrieve path and prediction from the prediction step
#     # WARNING: Prone to race conditions with multiple users. See warning above.
#     if 'LAST_MRI_PATH' not in app.config or 'LAST_MRI_PRED_IDX' not in app.config:
#          return jsonify({"error": "Please run prediction first or file path not found."}), 400
#
#     path = app.config['LAST_MRI_PATH']
#     # Use the ensemble prediction index for consistency if desired,
#     # or predict again for each model (current approach uses individual model preds)
#     # ensemble_pred_idx = app.config['LAST_MRI_PRED_IDX'] # Example if using ensemble index
#
#     try:
#         _, img_raw = preprocess_image(path) # Need the raw-like image for explanations
#
#         # Get individual predictions for explanation targeting
#         img_input, _ = preprocess_image(path) # Need input tensor for prediction
#         pred1_idx = np.argmax(mri_densenet.predict(img_input)[0])
#         pred2_idx = np.argmax(mri_resnet.predict(img_input)[0])
#
#         # Use ThreadPoolExecutor to run explanations in parallel
#         with ThreadPoolExecutor(max_workers=2) as executor:
#             # Submit tasks: args=(image, model, predicted_class_index, layer_name)
#             future1 = executor.submit(generate_explanations, img_raw, mri_densenet, pred1_idx, 'conv5_block32_concat')
#             future2 = executor.submit(generate_explanations, img_raw, mri_resnet, pred2_idx, 'conv5_block3_out')
#
#             # Get results
#             densenet_output = future1.result()
#             resnet_output = future2.result()
#
#         # Structure the response
#         response = {
#             "gradcam": {"densenet": densenet_output.get("gradcam"), "resnet": resnet_output.get("gradcam")},
#             "ig": {"densenet": densenet_output.get("ig"), "resnet": resnet_output.get("ig")},
#             "lime": {"densenet": densenet_output.get("lime"), "resnet": resnet_output.get("lime")}
#         }
#         return jsonify(response)
#
#     except FileNotFoundError:
#          return jsonify({"error": f"Image file not found at {path}. Please predict again."}), 404
#     except ValueError as ve:
#          return jsonify({"error": f"Image processing error: {str(ve)}"}), 400
#     except Exception as e:
#         import traceback
#         print(traceback.format_exc())
#         return jsonify({"error": f"Explanation failed: {str(e)}"}), 500
#
#
# # --- (predict_pet remains similar, add path storage) ---
# @app.route('/predict_pet', methods=['POST'])
# def predict_pet():
#     if 'file' not in request.files:
#         return jsonify({"error": "No file part"}), 400
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({"error": "No selected file"}), 400
#
#     try:
#         filename = secure_filename(file.filename)
#         path = os.path.join(UPLOAD_FOLDER, filename)
#         file.save(path)
#
#         img_input, _ = preprocess_image(path)
#         pred = pet_model.predict(img_input)[0]
#         pred_idx = np.argmax(pred)
#         pred_label = CLASSES[pred_idx]
#         confidence = round(float(pred[pred_idx]) * 100, 2)
#         all_confidences = {CLASSES[i]: round(float(pred[i] * 100), 2) for i in range(len(CLASSES))}
#
#         # Store path and prediction index
#         app.config['LAST_PET_PATH'] = path
#         app.config['LAST_PET_PRED_IDX'] = pred_idx
#
#         return jsonify({
#             "predicted_label": pred_label,
#             "confidence": confidence,
#             "actual_label": "Unknown",
#             "all_confidences": all_confidences
#         })
#     except ValueError as ve:
#          return jsonify({"error": f"Image processing error: {str(ve)}"}), 400
#     except Exception as e:
#         import traceback
#         print(traceback.format_exc())
#         return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
#
# # --- (explain_pet remains similar, uses generate_explanations) ---
# @app.route('/explain_pet', methods=['GET'])
# def explain_pet():
#     if 'LAST_PET_PATH' not in app.config or 'LAST_PET_PRED_IDX' not in app.config:
#          return jsonify({"error": "Please run PET prediction first or file path not found."}), 400
#
#     path = app.config['LAST_PET_PATH']
#     pred_class = app.config['LAST_PET_PRED_IDX']
#
#     try:
#         _, img_raw = preprocess_image(path)
#
#         # Generate explanations for the single PET model
#         pet_output = generate_explanations(img_raw, pet_model, pred_class, 'block5_conv4') # Ensure layer name is correct for VGG
#
#         return jsonify(pet_output) # Return directly as it's just one model
#
#     except FileNotFoundError:
#          return jsonify({"error": f"Image file not found at {path}. Please predict again."}), 404
#     except ValueError as ve:
#          return jsonify({"error": f"Image processing error: {str(ve)}"}), 400
#     except Exception as e:
#         import traceback
#         print(traceback.format_exc())
#         return jsonify({"error": f"Explanation failed: {str(e)}"}), 500
#
# # --- (predict_both remains similar, add path storage) ---
# @app.route('/predict_both', methods=['POST'])
# def predict_both():
#     if 'mri' not in request.files or 'pet' not in request.files:
#         return jsonify({"error": "Missing 'mri' or 'pet' file part"}), 400
#
#     mri_file = request.files['mri']
#     pet_file = request.files['pet']
#
#     if mri_file.filename == '' or pet_file.filename == '':
#         return jsonify({"error": "No selected file for MRI or PET"}), 400
#
#     try:
#         mri_filename = secure_filename("mri_" + mri_file.filename)
#         pet_filename = secure_filename("pet_" + pet_file.filename)
#         mri_path = os.path.join(UPLOAD_FOLDER, mri_filename)
#         pet_path = os.path.join(UPLOAD_FOLDER, pet_filename)
#         mri_file.save(mri_path)
#         pet_file.save(pet_path)
#
#         mri_input, _ = preprocess_image(mri_path)
#         pet_input, _ = preprocess_image(pet_path)
#
#         mri_pred1 = mri_densenet.predict(mri_input)[0]
#         mri_pred2 = mri_resnet.predict(mri_input)[0]
#         mri_ensemble = (mri_pred1 + mri_pred2) / 2
#         pet_pred = pet_model.predict(pet_input)[0]
#
#         final_probs = (mri_ensemble + pet_pred) / 2
#         fused_pred_idx = np.argmax(final_probs)
#         mri_pred_idx = np.argmax(mri_ensemble)
#         pet_pred_idx = np.argmax(pet_pred)
#
#         # Store paths and indices
#         app.config['LAST_BOTH_MRI_PATH'] = mri_path
#         app.config['LAST_BOTH_PET_PATH'] = pet_path
#         app.config['LAST_BOTH_MRI_DENSE_IDX'] = np.argmax(mri_pred1) # Store individual for explanation
#         app.config['LAST_BOTH_MRI_RES_IDX'] = np.argmax(mri_pred2) # Store individual for explanation
#         app.config['LAST_BOTH_PET_IDX'] = pet_pred_idx
#
#         return jsonify({
#             "fused": {
#                 "predicted_label": CLASSES[fused_pred_idx],
#                 "confidence": round(float(final_probs[fused_pred_idx]) * 100, 2),
#                 "all_confidences": {CLASSES[i]: round(float(final_probs[i] * 100), 2) for i in range(len(CLASSES))}
#             },
#             "mri": {
#                 "predicted_label": CLASSES[mri_pred_idx],
#                 "confidence": round(float(np.max(mri_ensemble)) * 100, 2),
#                 "all_confidences": {CLASSES[i]: round(float(mri_ensemble[i] * 100), 2) for i in range(len(CLASSES))}
#             },
#             "pet": {
#                 "predicted_label": CLASSES[pet_pred_idx],
#                 "confidence": round(float(np.max(pet_pred)) * 100, 2),
#                 "all_confidences": {CLASSES[i]: round(float(pet_pred[i] * 100), 2) for i in range(len(CLASSES))}
#             }
#         })
#
#     except ValueError as ve:
#          return jsonify({"error": f"Image processing error: {str(ve)}"}), 400
#     except Exception as e:
#         import traceback
#         print(traceback.format_exc())
#         return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
#
#
# # --- UPDATED /explain_both with Parallelism ---
# @app.route('/explain_both', methods=['GET'])
# def explain_both():
#     if ('LAST_BOTH_MRI_PATH' not in app.config or
#         'LAST_BOTH_PET_PATH' not in app.config or
#         'LAST_BOTH_MRI_DENSE_IDX' not in app.config or
#         'LAST_BOTH_MRI_RES_IDX' not in app.config or
#         'LAST_BOTH_PET_IDX' not in app.config):
#         return jsonify({"error": "Please run 'predict_both' first or required data not found."}), 400
#
#     mri_path = app.config['LAST_BOTH_MRI_PATH']
#     pet_path = app.config['LAST_BOTH_PET_PATH']
#     mri_pred1_idx = app.config['LAST_BOTH_MRI_DENSE_IDX']
#     mri_pred2_idx = app.config['LAST_BOTH_MRI_RES_IDX']
#     pet_pred_idx = app.config['LAST_BOTH_PET_IDX']
#
#     try:
#         _, mri_raw = preprocess_image(mri_path)
#         _, pet_raw = preprocess_image(pet_path)
#
#         # Use ThreadPoolExecutor for all 3 explanations
#         with ThreadPoolExecutor(max_workers=3) as executor:
#              # Submit MRI tasks
#             future_mri_dense = executor.submit(generate_explanations, mri_raw, mri_densenet, mri_pred1_idx, 'conv5_block32_concat')
#             future_mri_resnet = executor.submit(generate_explanations, mri_raw, mri_resnet, mri_pred2_idx, 'conv5_block3_out')
#             # Submit PET task
#             future_pet = executor.submit(generate_explanations, pet_raw, pet_model, pet_pred_idx, 'block5_conv4')
#
#             # Get results
#             mri_dense_out = future_mri_dense.result()
#             mri_resnet_out = future_mri_resnet.result()
#             pet_out = future_pet.result()
#
#         # Structure the response
#         return jsonify({
#             "mri": {
#                 "densenet": mri_dense_out,
#                 "resnet": mri_resnet_out
#             },
#             "pet": pet_out
#         })
#
#     except FileNotFoundError:
#          return jsonify({"error": "MRI or PET image file not found. Please predict again."}), 404
#     except ValueError as ve:
#          return jsonify({"error": f"Image processing error: {str(ve)}"}), 400
#     except Exception as e:
#         import traceback
#         print(traceback.format_exc())
#         return jsonify({"error": f"Explanation failed: {str(e)}"}), 500
#
#
# if __name__ == '__main__':
#     # Important: Use a production-ready WSGI server like gunicorn or waitress
#     # instead of Flask's built-in server for anything beyond development.
#     # Example: waitress-serve --host 0.0.0.0 --port 5000 your_app_module:app
#     # For threading to be effective with CPU-bound tasks outside TF GIL release,
#     # you might need multiple workers with gunicorn/waitress.
#     app.run(host="0.0.0.0", port=5000, debug=False, threaded=True) # Set debug=False for production/performance test, threaded=True helps

# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import os
# import numpy as np
# import tensorflow as tf
# import cv2
# from werkzeug.utils import secure_filename
# from lime import lime_image
# from skimage.segmentation import mark_boundaries
# import base64
# import matplotlib.cm as cm
# from concurrent.futures import ThreadPoolExecutor # Import ThreadPoolExecutor
# import traceback # For detailed error logging
#
# app = Flask(__name__)
# CORS(app)
#
# # --- Paths and Constants ---
# MRI_MODEL_PATH_DENSENET = "D:/SAI/MRI_DenseNet201_CBAM_best.h5"
# MRI_MODEL_PATH_RESNET = "D:/SAI/MRI_ResNet50_CBAM_E.h5"
# PET_MODEL_PATH = "D:/SAI/PET_VGG19_CBAM_Enhanced.h5"
# UPLOAD_FOLDER = "uploads"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# IMG_SIZE = 128
# CLASSES = ['AD', 'CN', 'EMCI', 'LMCI']
#
# # --- Load Models ---
# # Consider adding error handling here if models might not load
# print("Loading models...")
# try:
#     mri_densenet = tf.keras.models.load_model(MRI_MODEL_PATH_DENSENET)
#     mri_resnet = tf.keras.models.load_model(MRI_MODEL_PATH_RESNET)
#     pet_model = tf.keras.models.load_model(PET_MODEL_PATH)
#     print("Models loaded successfully.")
# except Exception as e:
#     print(f"FATAL ERROR: Could not load models. {e}")
#     # Optionally exit or raise the exception depending on desired behavior
#     # exit()
#
#
# # --- Helper Functions (Keep Original Versions) ---
# def encode_image_to_base64(image):
#     # Ensure image is uint8 before encoding
#     if image.dtype != np.uint8:
#         # Handle potential float images from LIME or overlays
#         if image.max() <= 1.0 and image.min() >= 0.0:
#              image = (image * 255).astype(np.uint8)
#         else:
#             # Attempt safe conversion if values might be outside 0-1 or 0-255
#             image = np.clip(image, 0, 255).astype(np.uint8)
#
#     _, buffer = cv2.imencode('.png', image)
#     return base64.b64encode(buffer).decode('utf-8')
#
# def apply_matplotlib_colormap(heatmap):
#     # Normalize heatmap to 0-1 range before applying colormap for consistency
#     heatmap_normalized = np.clip(heatmap, 0, 1)
#     cmap = cm.get_cmap('jet')
#     colored_heatmap = cmap(heatmap_normalized)[:, :, :3] # Take only RGB
#     return (colored_heatmap * 255).astype(np.uint8)
#
# def preprocess_image(image_path):
#     img = cv2.imread(image_path, cv2.IMREAD_COLOR)
#     if img is None:
#         # More specific error message
#         raise ValueError(f"Cannot load image or image not found at path: {image_path}")
#     # Ensure input is 3 channel BGR
#     if len(img.shape) == 2: # Grayscale
#         img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#     elif img.shape[2] == 4: # RGBA
#         img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
#
#     img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
#     # Return both the tensor input (normalized) and the raw-like image (resized, 0-1 float) for explanations
#     img_normalized_float = img_resized.astype(np.float32) / 255.0
#     return np.expand_dims(img_normalized_float, axis=0), img_normalized_float
#
# def get_brain_mask(image_float_01, threshold=0.1):
#     # Expects image input in float 0-1 range
#     if len(image_float_01.shape) == 3 and image_float_01.shape[2] == 3:
#          gray_image = np.mean(image_float_01, axis=-1)
#     elif len(image_float_01.shape) == 2:
#          gray_image = image_float_01
#     else:
#          # Fallback or raise error if unexpected shape
#          print(f"Warning: Unexpected image shape for brain mask: {image_float_01.shape}. Using first channel.")
#          gray_image = image_float_01[:,:,0] # Assume first channel if unsure
#
#     mask = (gray_image > threshold).astype(np.float32)
#     # Return mask expanded to 3 dims for potential broadcasting needs
#     # return np.stack([mask]*3, axis=-1) # -> No, return 2D mask is better
#     return mask
#
# # --- Explanation Functions (Keep Original Parameters) ---
#
# def integrated_gradients(image_float_01, model, class_index, steps=50):
#     # Expects image_float_01 (shape H, W, C, range 0-1)
#     image = image_float_01.astype(np.float32)
#     baseline = np.zeros_like(image).astype(np.float32)
#     interpolated_images = [baseline + (float(i) / steps) * (image - baseline) for i in range(steps + 1)]
#     interpolated_images_tensor = tf.convert_to_tensor(np.array(interpolated_images), dtype=tf.float32)
#
#     with tf.GradientTape() as tape:
#         tape.watch(interpolated_images_tensor)
#         preds = model(interpolated_images_tensor)
#         # Ensure class_index is valid
#         if class_index >= preds.shape[1]:
#             print(f"Warning: IG class_index {class_index} out of bounds for prediction shape {preds.shape}. Using index 0.")
#             class_index = 0
#         target = preds[:, class_index]
#
#     grads = tape.gradient(target, interpolated_images_tensor)
#     if grads is None:
#          print(f"Warning: IG Gradients are None for class {class_index}. Returning zeros.")
#          return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
#
#     grads = grads.numpy()
#     avg_grads = np.mean(grads, axis=0)
#     integrated_grads = (image - baseline) * avg_grads
#     attributions = np.mean(np.abs(integrated_grads), axis=-1) # Aggregate channels
#     attributions = np.nan_to_num(attributions) # Handle potential NaNs
#
#     # Normalize 0-1
#     min_val, max_val = np.min(attributions), np.max(attributions)
#     if max_val > min_val:
#         attributions = (attributions - min_val) / (max_val - min_val + 1e-8)
#     else:
#         attributions = np.zeros_like(attributions) # Avoid division by zero if flat
#
#     # Mask out background AFTER normalization
#     brain_mask = get_brain_mask(image_float_01)
#     return attributions * brain_mask # Return 2D heatmap
#
#
# def grad_cam(image_float_01, model, class_idx, layer_name):
#     # Expects image_float_01 (shape H, W, C, range 0-1)
#     image_tensor = tf.convert_to_tensor(image_float_01[np.newaxis, ...], dtype=tf.float32)
#
#     try:
#         target_layer = model.get_layer(layer_name)
#     except ValueError:
#         print(f"Warning: Layer '{layer_name}' not found directly for model {model.name}. Searching...")
#         target_layer = None
#         # Simple search for layers containing the name (can be improved)
#         for layer in reversed(model.layers):
#              if layer_name in layer.name:
#                   target_layer = layer
#                   print(f"Using layer: {target_layer.name}")
#                   break
#         if target_layer is None:
#              # Fallback: Find last Conv layer
#              for layer in reversed(model.layers):
#                   if isinstance(layer, tf.keras.layers.Conv2D):
#                        target_layer = layer
#                        print(f"Fallback: Using last Conv2D layer: {target_layer.name} for model {model.name}")
#                        break
#         if target_layer is None:
#             raise ValueError(f"Cannot find suitable layer like '{layer_name}' or any Conv layer for GradCAM in model {model.name}")
#
#     grad_model = tf.keras.models.Model([model.inputs], [target_layer.output, model.output])
#
#     with tf.GradientTape() as tape:
#         conv_outputs, predictions = grad_model(image_tensor)
#         if class_idx >= predictions.shape[1]:
#              print(f"Warning: GradCAM class_idx {class_idx} out of bounds for prediction shape {predictions.shape}. Using index 0.")
#              class_idx = 0
#         loss = predictions[:, class_idx]
#
#     grads = tape.gradient(loss, conv_outputs)
#     if grads is None:
#          print(f"Warning: GradCAM Gradients are None for layer {target_layer.name}, class {class_idx}. Returning zeros.")
#          return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
#
#     # Handle potential dimension issues if grads or conv_outputs shape is unexpected
#     # Assuming grads shape is (batch, h, w, c) and conv_outputs is (batch, h, w, c)
#     pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)) # Global Average Pooling G-CAM style often works well
#     # Original approach: pooled_grads = tf.reduce_mean(grads, axis=(1, 2)) # Pool spatial dims H, W
#
#     # Multiply feature map by pooled gradients (broadcasting)
#     # Ensure conv_outputs is squeezed if batch dim was 1
#     conv_outputs_squeezed = tf.squeeze(conv_outputs, axis=0)
#     heatmap = tf.reduce_sum(tf.multiply(conv_outputs_squeezed, pooled_grads), axis=-1) # Sum over channels
#     # heatmap = tf.reduce_mean(tf.multiply(pooled_grads[..., tf.newaxis, tf.newaxis, :], conv_outputs), axis=-1).numpy().squeeze() # Original approach
#
#     heatmap = heatmap.numpy()
#     heatmap = np.maximum(heatmap, 0) # ReLU
#
#     # Normalize 0-1
#     max_val = np.max(heatmap)
#     if max_val > 0:
#         heatmap /= max_val
#     else:
#         heatmap = np.zeros_like(heatmap) # Handle case where heatmap is all zeros
#
#     # Resize to original image size
#     return cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
#
#
# def lime_explainer(image_float_01, model):
#     # Expects image_float_01 (shape H, W, C, range 0-1)
#     explainer = lime_image.LimeImageExplainer()
#
#     # LIME expects a function that takes a batch of images (N, H, W, C)
#     # with pixel values in the range [0, 1] (or 0-255 depending on model)
#     # and returns predictions (N, num_classes).
#     def predict_fn_for_lime(images_double):
#         # LIME passes images as numpy array of doubles in range [0, 1]
#         # Convert to float32 for the model prediction
#         images_float32 = images_double.astype(np.float32)
#         return model.predict(images_float32)
#
#     explanation = explainer.explain_instance(
#         image_float_01.astype('double'), # LIME prefers double
#         predict_fn_for_lime,
#         top_labels=len(CLASSES),
#         hide_color=0,
#         num_samples=500, # <<< KEEPING ORIGINAL PARAMETER as requested
#         random_seed=42 # Add for reproducibility
#     )
#
#     # Predict once outside LIME to get the top class reliably
#     # Use the same input format as explanation for consistency
#     preds = model.predict(image_float_01[np.newaxis, ...].astype(np.float32))
#     pred_class = np.argmax(preds[0])
#     if pred_class >= len(CLASSES):
#         print(f"Warning: LIME predicted class {pred_class} out of bounds. Using 0.")
#         pred_class = 0
#
#     temp, mask = explanation.get_image_and_mask(
#         pred_class,
#         positive_only=True,
#         num_features=8, # <<< KEEPING ORIGINAL PARAMETER as requested
#         hide_rest=False,
#         min_weight=0.01 # Optional: filter small features
#     )
#     # 'temp' is the image used by LIME (should be close to input), float 0-1
#     # 'mask' highlights the superpixels for the explanation
#
#     # Use the original image for overlaying boundaries for clarity
#     img_uint8 = (image_float_01 * 255).astype(np.uint8)
#     mask_bool = mask > 0 # Ensure mask is boolean
#     # Green boundaries BGR=(0, 255, 0) -> color=(0, 1, 0) for mark_boundaries
#     boundaries = mark_boundaries(img_uint8, mask_bool, color=(0, 1, 0), mode='thick')
#
#     # boundaries is float 0-1 image with BGR channels
#     return boundaries
#
#
# # --- Centralized Explanation Generation Function (called by threads) ---
# def generate_explanations(image_raw_float, model, pred_class, layer_name=None):
#     """Generates GradCAM, IG, and LIME explanations for a single model.
#        Expects image_raw_float (H, W, C) with range 0-1.
#        Returns a dictionary with base64 encoded explanation images.
#     """
#     results = {"gradcam": None, "ig": None, "lime": None}
#     model_name = model.name if hasattr(model, 'name') else 'unknown_model'
#     img_uint8_bgr = (image_raw_float * 255).astype(np.uint8) # For overlays
#
#     print(f"Starting explanations for {model_name}, class {pred_class}...")
#
#     # --- Grad-CAM ---
#     if layer_name:
#         try:
#             print(f"  Generating Grad-CAM for {model_name}...")
#             grad = grad_cam(image_raw_float, model, pred_class, layer_name)
#             # Apply brain mask *before* colormap and overlay
#             grad_masked = np.clip(grad * get_brain_mask(image_raw_float), 0, 1)
#             grad_heatmap_bgr = apply_matplotlib_colormap(grad_masked) # BGR output from colormap
#             grad_overlay = cv2.addWeighted(img_uint8_bgr, 0.6, grad_heatmap_bgr, 0.4, 0)
#             results["gradcam"] = encode_image_to_base64(cv2.cvtColor(grad_overlay, cv2.COLOR_BGR2RGB)) # Encode as RGB
#             print(f"  Grad-CAM done for {model_name}.")
#         except Exception as e:
#             print(f"  ERROR generating Grad-CAM for {model_name}: {e}")
#             traceback.print_exc() # Print detailed error
#     else:
#         print(f"  Skipping Grad-CAM for {model_name} (no layer name provided).")
#
#     # --- Integrated Gradients ---
#     try:
#         print(f"  Generating Integrated Gradients for {model_name}...")
#         ig = integrated_gradients(image_raw_float, model, pred_class, steps=50) # Use original steps=50
#         # Optional: Apply gamma correction if needed (kept from original code)
#         ig = ig ** 0.5
#         ig_heatmap_bgr = apply_matplotlib_colormap(ig) # IG includes brain mask, output BGR
#         # Stronger overlay weight (kept from original code)
#         ig_overlay = cv2.addWeighted(img_uint8_bgr, 0.4, ig_heatmap_bgr, 0.6, 0)
#         results["ig"] = encode_image_to_base64(cv2.cvtColor(ig_overlay, cv2.COLOR_BGR2RGB)) # Encode as RGB
#         print(f"  Integrated Gradients done for {model_name}.")
#     except Exception as e:
#         print(f"  ERROR generating Integrated Gradients for {model_name}: {e}")
#         traceback.print_exc()
#
#     # --- LIME ---
#     try:
#         print(f"  Generating LIME for {model_name}...")
#         # LIME returns a float image (0-1) with boundaries already drawn (BGR format from mark_boundaries)
#         lime_img_float_bgr = lime_explainer(image_raw_float, model)
#         # Convert to uint8 RGB for encoding
#         lime_img_uint8_rgb = cv2.cvtColor((lime_img_float_bgr * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
#         results["lime"] = encode_image_to_base64(lime_img_uint8_rgb)
#         print(f"  LIME done for {model_name}.")
#     except Exception as e:
#         print(f"  ERROR generating LIME for {model_name}: {e}")
#         traceback.print_exc()
#
#     print(f"Finished explanations for {model_name}.")
#     return results
#
# # --- Routes ---
#
# @app.route('/')
# def home():
#     return "XAI MRI & PET API for Alzheimer's detection is live!"
#
# @app.route('/predict_mri', methods=['POST'])
# def predict_mri():
#     if 'file' not in request.files:
#         return jsonify({"error": "No file part"}), 400
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({"error": "No selected file"}), 400
#
#     try:
#         filename = secure_filename(file.filename)
#         # Ensure unique filenames if concurrent users might upload same name
#         # Could add timestamp or UUID, but keeping simple for now
#         path = os.path.join(UPLOAD_FOLDER, filename)
#         file.save(path)
#         print(f"MRI File saved to: {path}")
#
#         img_input_tensor, _ = preprocess_image(path) # Need tensor for prediction
#
#         # Get predictions
#         pred1_probs = mri_densenet.predict(img_input_tensor)[0]
#         pred2_probs = mri_resnet.predict(img_input_tensor)[0]
#         ensemble_probs = (pred1_probs + pred2_probs) / 2
#
#         # Get indices for explanations later
#         pred1_idx = int(np.argmax(pred1_probs)) # Ensure integer type
#         pred2_idx = int(np.argmax(pred2_probs))
#         ensemble_pred_idx = int(np.argmax(ensemble_probs))
#
#         # Store path and necessary prediction indices in app.config
#         # WARNING: NOT THREAD SAFE / PRODUCTION READY - Prone to race conditions
#         app.config['LAST_MRI_PATH'] = path
#         app.config['LAST_MRI_DENSE_IDX'] = pred1_idx
#         app.config['LAST_MRI_RES_IDX'] = pred2_idx
#         print(f"Stored MRI path: {path}, DenseIdx: {pred1_idx}, ResIdx: {pred2_idx}")
#
#         # Prepare response
#         return jsonify({
#             "predicted_label": CLASSES[ensemble_pred_idx],
#             "confidence": round(float(ensemble_probs[ensemble_pred_idx]) * 100, 2),
#             "actual_label": "Unknown", # Assuming no ground truth provided
#             "all_confidences": {CLASSES[i]: round(float(ensemble_probs[i] * 100), 2) for i in range(len(CLASSES))},
#             # Optionally return the filename for robust explain calls
#             "filename": filename
#         })
#     except ValueError as ve:
#          print(f"Error during MRI prediction preprocessing: {ve}")
#          traceback.print_exc()
#          return jsonify({"error": f"Image processing error: {str(ve)}"}), 400
#     except Exception as e:
#         print(f"Error during MRI prediction: {e}")
#         traceback.print_exc()
#         return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
#
#
# @app.route('/explain_mri', methods=['GET'])
# def explain_mri():
#     # --- Retrieve data stored from prediction step ---
#     # WARNING: Using app.config is NOT thread-safe for concurrent users!
#     # A better approach: Get filename from request args (e.g., /explain_mri?filename=...)
#     # filename = request.args.get('filename')
#     # if not filename:
#     #    return jsonify({"error": "Missing 'filename' parameter in request."}), 400
#     # path = os.path.join(UPLOAD_FOLDER, secure_filename(filename))
#     # TODO: Need a way to get pred_idx if using filename param (e.g., re-predict or store in a DB/cache)
#
#     # Using app.config approach for this example:
#     if ('LAST_MRI_PATH' not in app.config or
#         'LAST_MRI_DENSE_IDX' not in app.config or
#         'LAST_MRI_RES_IDX' not in app.config):
#          return jsonify({"error": "Prediction data not found. Please run prediction first."}), 400
#
#     path = app.config.get('LAST_MRI_PATH')
#     pred1_idx = app.config.get('LAST_MRI_DENSE_IDX')
#     pred2_idx = app.config.get('LAST_MRI_RES_IDX')
#     print(f"Explaining MRI for path: {path}, DenseIdx: {pred1_idx}, ResIdx: {pred2_idx}")
#     # --- End of data retrieval ---
#
#     try:
#         if not os.path.exists(path):
#              return jsonify({"error": f"Image file not found at {path}. It might have been deleted or prediction was for a different file."}), 404
#
#         _, img_raw_float = preprocess_image(path) # Need the raw-like float image (0-1)
#
#         # Use ThreadPoolExecutor to run explanations in parallel
#         results = {}
#         # max_workers=2 limits concurrency to 2 threads for this endpoint
#         with ThreadPoolExecutor(max_workers=2, thread_name_prefix='MRIExplain') as executor:
#             print("Submitting MRI explanation tasks to thread pool...")
#             # Submit tasks: args=(image_raw_float, model, predicted_class_index, layer_name)
#             future1 = executor.submit(generate_explanations, img_raw_float, mri_densenet, pred1_idx, 'conv5_block32_concat')
#             future2 = executor.submit(generate_explanations, img_raw_float, mri_resnet, pred2_idx, 'conv5_block3_out') # Ensure layer name is correct
#
#             # Get results (blocks until tasks are done)
#             print("Waiting for MRI explanation tasks to complete...")
#             densenet_output = future1.result()
#             resnet_output = future2.result()
#             print("MRI explanation tasks finished.")
#
#         # Structure the response
#         response = {
#             "gradcam": {"densenet": densenet_output.get("gradcam"), "resnet": resnet_output.get("gradcam")},
#             "ig": {"densenet": densenet_output.get("ig"), "resnet": resnet_output.get("ig")},
#             "lime": {"densenet": densenet_output.get("lime"), "resnet": resnet_output.get("lime")}
#         }
#         return jsonify(response)
#
#     except ValueError as ve:
#          print(f"Error during MRI explanation preprocessing: {ve}")
#          traceback.print_exc()
#          return jsonify({"error": f"Image processing error: {str(ve)}"}), 400
#     except FileNotFoundError: # Catch potentially missed file check
#          print(f"Error: Image file disappeared before explanation: {path}")
#          return jsonify({"error": f"Image file not found at {path}. Please predict again."}), 404
#     except Exception as e:
#         print(f"Error during MRI explanation: {e}")
#         traceback.print_exc()
#         return jsonify({"error": f"Explanation failed: {str(e)}"}), 500
#
#
# @app.route('/predict_pet', methods=['POST'])
# def predict_pet():
#     if 'file' not in request.files:
#         return jsonify({"error": "No file part"}), 400
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({"error": "No selected file"}), 400
#
#     try:
#         filename = secure_filename(file.filename)
#         path = os.path.join(UPLOAD_FOLDER, filename)
#         file.save(path)
#         print(f"PET File saved to: {path}")
#
#         img_input_tensor, _ = preprocess_image(path)
#         pred_probs = pet_model.predict(img_input_tensor)[0]
#         pred_idx = int(np.argmax(pred_probs))
#
#         # Store path and prediction index for explanation
#         # WARNING: NOT THREAD SAFE / PRODUCTION READY
#         app.config['LAST_PET_PATH'] = path
#         app.config['LAST_PET_PRED_IDX'] = pred_idx
#         print(f"Stored PET path: {path}, PredIdx: {pred_idx}")
#
#
#         return jsonify({
#             "predicted_label": CLASSES[pred_idx],
#             "confidence": round(float(pred_probs[pred_idx]) * 100, 2),
#             "actual_label": "Unknown",
#             "all_confidences": {CLASSES[i]: round(float(pred_probs[i] * 100), 2) for i in range(len(CLASSES))},
#             "filename": filename # Return filename
#         })
#     except ValueError as ve:
#          print(f"Error during PET prediction preprocessing: {ve}")
#          traceback.print_exc()
#          return jsonify({"error": f"Image processing error: {str(ve)}"}), 400
#     except Exception as e:
#         print(f"Error during PET prediction: {e}")
#         traceback.print_exc()
#         return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
#
# @app.route('/explain_pet', methods=['GET'])
# def explain_pet():
#     # --- Retrieve data stored from prediction step ---
#     # WARNING: Using app.config is NOT thread-safe for concurrent users!
#     # See /explain_mri for comments on using filename parameter instead.
#     if ('LAST_PET_PATH' not in app.config or
#         'LAST_PET_PRED_IDX' not in app.config):
#          return jsonify({"error": "PET Prediction data not found. Please run prediction first."}), 400
#
#     path = app.config.get('LAST_PET_PATH')
#     pred_class = app.config.get('LAST_PET_PRED_IDX')
#     print(f"Explaining PET for path: {path}, PredClass: {pred_class}")
#     # --- End of data retrieval ---
#
#     try:
#         if not os.path.exists(path):
#              return jsonify({"error": f"Image file not found at {path}. Please predict again."}), 404
#
#         _, img_raw_float = preprocess_image(path)
#
#         # Generate explanations for the single PET model (no thread pool needed here, but use same func)
#         # Ensure the layer name 'block5_conv4' is correct for your PET_VGG19_CBAM_Enhanced model
#         print("Generating PET explanations...")
#         pet_output = generate_explanations(img_raw_float, pet_model, pred_class, 'block5_conv4')
#         print("PET explanations finished.")
#
#         return jsonify(pet_output) # Return explanation dict directly
#
#     except ValueError as ve:
#          print(f"Error during PET explanation preprocessing: {ve}")
#          traceback.print_exc()
#          return jsonify({"error": f"Image processing error: {str(ve)}"}), 400
#     except FileNotFoundError:
#          print(f"Error: Image file disappeared before explanation: {path}")
#          return jsonify({"error": f"Image file not found at {path}. Please predict again."}), 404
#     except Exception as e:
#         print(f"Error during PET explanation: {e}")
#         traceback.print_exc()
#         return jsonify({"error": f"Explanation failed: {str(e)}"}), 500
#
#
# @app.route('/predict_both', methods=['POST'])
# def predict_both():
#     if 'mri' not in request.files or 'pet' not in request.files:
#         return jsonify({"error": "Missing 'mri' or 'pet' file part"}), 400
#
#     mri_file = request.files['mri']
#     pet_file = request.files['pet']
#
#     if mri_file.filename == '' or pet_file.filename == '':
#         return jsonify({"error": "No selected file for MRI or PET"}), 400
#
#     try:
#         # Use potentially more unique names
#         mri_filename = secure_filename("mri_" + mri_file.filename)
#         pet_filename = secure_filename("pet_" + pet_file.filename)
#         mri_path = os.path.join(UPLOAD_FOLDER, mri_filename)
#         pet_path = os.path.join(UPLOAD_FOLDER, pet_filename)
#         mri_file.save(mri_path)
#         pet_file.save(pet_path)
#         print(f"Both files saved: MRI={mri_path}, PET={pet_path}")
#
#
#         mri_input_tensor, _ = preprocess_image(mri_path)
#         pet_input_tensor, _ = preprocess_image(pet_path)
#
#         # Predictions
#         mri_pred1_probs = mri_densenet.predict(mri_input_tensor)[0]
#         mri_pred2_probs = mri_resnet.predict(mri_input_tensor)[0]
#         mri_ensemble_probs = (mri_pred1_probs + mri_pred2_probs) / 2
#         pet_pred_probs = pet_model.predict(pet_input_tensor)[0]
#
#         # Fusion
#         final_probs = (mri_ensemble_probs + pet_pred_probs) / 2
#
#         # Get indices
#         fused_pred_idx = int(np.argmax(final_probs))
#         mri_ensemble_pred_idx = int(np.argmax(mri_ensemble_probs))
#         pet_pred_idx = int(np.argmax(pet_pred_probs))
#         # Also get individual MRI model indices for explanation
#         mri_dense_idx = int(np.argmax(mri_pred1_probs))
#         mri_res_idx = int(np.argmax(mri_pred2_probs))
#
#
#         # Store paths and indices needed for explanation
#         # WARNING: NOT THREAD SAFE / PRODUCTION READY
#         app.config['LAST_BOTH_MRI_PATH'] = mri_path
#         app.config['LAST_BOTH_PET_PATH'] = pet_path
#         app.config['LAST_BOTH_MRI_DENSE_IDX'] = mri_dense_idx
#         app.config['LAST_BOTH_MRI_RES_IDX'] = mri_res_idx
#         app.config['LAST_BOTH_PET_IDX'] = pet_pred_idx
#         print(f"Stored Both paths: MRI={mri_path}, PET={pet_path}")
#         print(f"Stored Both indices: Dense={mri_dense_idx}, Res={mri_res_idx}, PET={pet_pred_idx}")
#
#
#         # Prepare response
#         return jsonify({
#             "fused": {
#                 "predicted_label": CLASSES[fused_pred_idx],
#                 "confidence": round(float(final_probs[fused_pred_idx]) * 100, 2),
#                 "all_confidences": {CLASSES[i]: round(float(final_probs[i] * 100), 2) for i in range(len(CLASSES))}
#             },
#             "mri": {
#                 "predicted_label": CLASSES[mri_ensemble_pred_idx],
#                 "confidence": round(float(np.max(mri_ensemble_probs)) * 100, 2),
#                 "all_confidences": {CLASSES[i]: round(float(mri_ensemble_probs[i] * 100), 2) for i in range(len(CLASSES))}
#             },
#             "pet": {
#                 "predicted_label": CLASSES[pet_pred_idx],
#                 "confidence": round(float(np.max(pet_pred_probs)) * 100, 2),
#                 "all_confidences": {CLASSES[i]: round(float(pet_pred_probs[i] * 100), 2) for i in range(len(CLASSES))}
#             },
#             "mri_filename": mri_filename, # Return filenames
#             "pet_filename": pet_filename
#         })
#
#     except ValueError as ve:
#          print(f"Error during Both prediction preprocessing: {ve}")
#          traceback.print_exc()
#          return jsonify({"error": f"Image processing error: {str(ve)}"}), 400
#     except Exception as e:
#         print(f"Error during Both prediction: {e}")
#         traceback.print_exc()
#         return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
#
#
# @app.route('/explain_both', methods=['GET'])
# def explain_both():
#     # --- Retrieve data stored from prediction step ---
#     # WARNING: Using app.config is NOT thread-safe for concurrent users!
#     # See /explain_mri for comments on using filename parameters instead.
#     required_keys = [
#         'LAST_BOTH_MRI_PATH', 'LAST_BOTH_PET_PATH',
#         'LAST_BOTH_MRI_DENSE_IDX', 'LAST_BOTH_MRI_RES_IDX', 'LAST_BOTH_PET_IDX'
#     ]
#     if not all(key in app.config for key in required_keys):
#          return jsonify({"error": "Prediction data not found for 'both'. Please run '/predict_both' first."}), 400
#
#     mri_path = app.config.get('LAST_BOTH_MRI_PATH')
#     pet_path = app.config.get('LAST_BOTH_PET_PATH')
#     mri_dense_idx = app.config.get('LAST_BOTH_MRI_DENSE_IDX')
#     mri_res_idx = app.config.get('LAST_BOTH_MRI_RES_IDX')
#     pet_pred_idx = app.config.get('LAST_BOTH_PET_IDX')
#     print(f"Explaining Both: MRI={mri_path} (Dense:{mri_dense_idx}, Res:{mri_res_idx}), PET={pet_path} (PET:{pet_pred_idx})")
#     # --- End of data retrieval ---
#
#     try:
#         if not os.path.exists(mri_path) or not os.path.exists(pet_path):
#              missing = []
#              if not os.path.exists(mri_path): missing.append(mri_path)
#              if not os.path.exists(pet_path): missing.append(pet_path)
#              return jsonify({"error": f"Image file(s) not found: {', '.join(missing)}. Please predict again."}), 404
#
#         _, mri_raw_float = preprocess_image(mri_path)
#         _, pet_raw_float = preprocess_image(pet_path)
#
#         # Use ThreadPoolExecutor for all 3 explanations
#         # max_workers=3 allows all three to potentially run concurrently
#         with ThreadPoolExecutor(max_workers=3, thread_name_prefix='BothExplain') as executor:
#             print("Submitting Both explanation tasks to thread pool...")
#             # Submit MRI tasks
#             future_mri_dense = executor.submit(generate_explanations, mri_raw_float, mri_densenet, mri_dense_idx, 'conv5_block32_concat')
#             future_mri_resnet = executor.submit(generate_explanations, mri_raw_float, mri_resnet, mri_res_idx, 'conv5_block3_out')
#             # Submit PET task
#             future_pet = executor.submit(generate_explanations, pet_raw_float, pet_model, pet_pred_idx, 'block5_conv4') # Ensure layer name is correct
#
#             # Get results
#             print("Waiting for Both explanation tasks to complete...")
#             mri_dense_out = future_mri_dense.result()
#             mri_resnet_out = future_mri_resnet.result()
#             pet_out = future_pet.result()
#             print("Both explanation tasks finished.")
#
#         # Structure the response
#         return jsonify({
#             "mri": {
#                 "densenet": mri_dense_out,
#                 "resnet": mri_resnet_out
#             },
#             "pet": pet_out
#         })
#
#     except ValueError as ve:
#          print(f"Error during Both explanation preprocessing: {ve}")
#          traceback.print_exc()
#          return jsonify({"error": f"Image processing error: {str(ve)}"}), 400
#     except FileNotFoundError: # Catch potentially missed file check
#          print(f"Error: Image file disappeared before explanation: MRI={mri_path}, PET={pet_path}")
#          return jsonify({"error": f"Image file not found. Please predict again."}), 404
#     except Exception as e:
#         print(f"Error during Both explanation: {e}")
#         traceback.print_exc()
#         return jsonify({"error": f"Explanation failed: {str(e)}"}), 500
#
#
# if __name__ == '__main__':
#     # Run with threaded=True to allow Flask to handle requests concurrently using threads.
#     # Set debug=False for performance testing/pseudo-production.
#     # For true production, use a WSGI server like gunicorn or waitress.
#     # Example: waitress-serve --host 0.0.0.0 --port 5000 your_module_name:app
#     print("Starting Flask app...")
#     app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)








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