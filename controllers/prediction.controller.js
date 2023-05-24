const tf = require('@tensorflow/tfjs-node');
const image = require('get-image-data');
const fs = require('fs');
var path = require('path');

const classes = ['rock', 'paper', 'scissors'];

exports.makePredictions = async (req, res, next) => {
    const imagePath = `./public/images/${req && req['filename']}`;
    try {
      const loadModel = async (img) => {
        const output = {};
        // laod model
        console.log('Loading.......')
        const model = await tf.node.loadSavedModel(path.join(__dirname,'..', 'SavedModels'));
        // classify
        // output.predictions = await model.predict(img).data();
        let predictions = await model.predict(img).data();
        // let pred_class = tf.argMax(predictions, axis=1)
        console.log('[My Data] ',predictions)
        predictions = Array.from(predictions)
                            .map((prob, idx) => {
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
                                console.log('[Mine]',prob)
                                return {class: cs[idx], probability: prob}
                            })
                            .sort((a, b) => b.probability - a.probability)[0];
        output.success = true;
        output.message = `Success.`;
        output.predictions = predictions;
        res.statusCode = 200;
        res.json(output);
      };
      await image(imagePath, async (err, imageData) => {
        try {
            const image = fs.readFileSync(imagePath);
            let tensor = tf.node.decodeImage(image);
            const resizedImage = tensor.resizeNearestNeighbor([150, 150]);
            const batchedImage = resizedImage.expandDims(0);
            const input = batchedImage.toFloat().div(tf.scalar(255));
            await loadModel(input);
            // delete image file
            fs.unlinkSync(imagePath, (error) => {
            if (error) {
                console.error(error);
            }
            });
        } catch (error) {
            res.status(500).json({message: "Internal Server Error!"});   
        }
      });
    } catch (error) {
      console.log(error)
    }
  };