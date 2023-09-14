const tf = require('@tensorflow/tfjs-node');
const image = require('get-image-data');
const fs = require('fs');
var path = require('path');

var cs = [
  'Apple => Apple_scab', 
  'Apple => Black_rot', 
  'Apple => Cedar_apple_rust',
  "Apple => healthy",
  "Blueberry => healthy",
  "Cherry_(including_sour) => Powdery_mildew",
  "Cherry_(including_sour) => healthy",
  "Corn_(maize) => Cercospora_leaf_spot Gray_leaf_spot",
  "Corn_(maize) => Common_rust_",
  "Corn_(maize) => Northern_Leaf_Blight",
  "Corn_(maize) => healthy",
  "Grape => Black_rot",
  "Grape => Esca_(Black_Measles)",
  "Grape => Leaf_blight_(Isariopsis_Leaf_Spot)",
  "Grape => healthy",
  "Orange => Haunglongbing_(Citrus_greening)",
  "Peach => Bacterial_spot",
  "Peach => healthy",
  "Pepper_bell => Bacterial_spot",
  "Pepper_bell => healthy",
  "Potato => Early_blight",
  "Potato => Late_blight",
  "Potato => healthy",
  "Raspberry => healthy",
  "Soybean => healthy",
  "Squash => Powdery_mildew",
  "Strawberry => Leaf_scorch",
  "Strawberry => healthy",
  "Tomato => Bacterial_spot",
  "Tomato => Early_blight",
  "Tomato => Late_blight",
  "Tomato => Leaf_Mold",
  "Tomato => Septoria_leaf_spot",
  "Tomato => Spider_mites Two-spotted_spider_mite",
  "Tomato => Target_Spot",
  "Tomato => Tomato_Yellow_Leaf_Curl_Virus",
  "Tomato => Tomato_mosaic_virus",
  "Tomato => healthy"
];

exports.makePredictions = async (req, res, next) => {
    const imagePath = `./public/images/${req && req['filename']}`;
    const modelPath = path.join(__dirname,'..', 'savedModels')
    try {
      const loadModel = async (img) => {
        const output = {};
        // laod model
        console.log('Loading.......')
        const model = await tf.node.loadSavedModel(modelPath);
        // classify
        // output.predictions = await model.predict(img).data();
        let predictions = await model.predict(img).data();
        // let predictions2 = await model.predict(img).data()
        // let pred_class = tf.argMax(predictions, axis=1)
        // console.log('[My Data] ',predictions)
        predictions = Array.from(predictions)
                            .map((prob, idx) => {
                                
                                // console.log('[Mine]',prob)
                                return {class: cs[idx], probability: prob, index: idx}
                            })
                            .sort((a, b) => b.probability - a.probability)[0];
        output.success = true;
        output.message = `Success.`;
        output.predictions = predictions;
        // output.index = 
        res.statusCode = 200;
        res.json(output);
      };

            // Function to resize the image while preserving aspect ratio
      async function resizeImageWithAspectRatio(image, targetSize) {
        const [originalHeight, originalWidth] = image.shape.slice(0, 2);

        // Calculate the aspect ratio of the original image
        const aspectRatio = originalWidth / originalHeight;
        // Set the desired maximum dimension
        const maxDimension = 500;

        // Calculate the scaling factor for resizing
        const scaleFactor = Math.min(maxDimension / originalWidth, maxDimension / originalHeight);

        // Calculate the new width and height
        const newWidth = Math.round(originalWidth * scaleFactor);
        const newHeight = Math.round(originalHeight * scaleFactor);
        // Calculate the new dimensions based on the target size and aspect ratio 
        console.log('[Data]', newWidth)
        // Resize the image while preserving aspect ratio
        const resizedImage = tf.image.resizeBilinear(image, [newHeight, newWidth]);

        return resizedImage;
      }
      await image(imagePath, async (err, imageData) => {
        try {
            const image = fs.readFileSync(imagePath);
            let tensor = tf.node.decodeImage(image);
            //test
            const targetSize = 256;
            const resizedImageRatio = await resizeImageWithAspectRatio(tensor, targetSize);
            console.log('[Hello]',resizedImageRatio.shape)
            const newHeight = resizedImageRatio.shape[0]
            const newWidth = resizedImageRatio.shape[1]
            const resizedImage = tensor.resizeNearestNeighbor([newHeight, newWidth]);
            const batchedImage = resizedImage.expandDims(0)
            const input = batchedImage.toFloat().div(tf.scalar(255));
            console.log('hello',input)
            await loadModel(input);
            // delete image file
            // fs.unlink(imagePath)
            fs.unlink(imagePath, (error) => {
            if (error) {
                console.error(error);
            }
            });
        } catch (error) {
          // console.log(error)
            res.status(500).json({message: "Internal Server Errors!"});   
        }
      });
    } catch (error) {
      console.log(error)
    }
  };