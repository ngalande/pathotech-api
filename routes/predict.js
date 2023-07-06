const express = require('express');
const multer = require('multer');
const controller = require('../controllers/prediction.controller');

const filename = `test-image-${Date.now().toString()}.jpg`;

// configure multer
const storage = multer.diskStorage({
  destination: (req, file, callback) => {
    callback(null, './public/images');
  },
  filename: (req, file, callback) => {
    callback(null, filename);
  },
});

function imageFileFilter(req, file, cb){
  // Allowed ext
  const filetypes = /jpeg|jpg|JPG|png|gif/;
  // Check ext
  const extname = file.originalname.match(filetypes);
  // Check mime
  const mimetype = filetypes.test(file.mimetype);

  if(mimetype && extname){
    return cb(null,true);
  } else {
    cb('Error: Images Only!');
  }
}

const upload = multer({ storage, fileFilter: imageFileFilter });
let uploadSingle = upload.single('file');
const router = express.Router();
router.post('/', (req, res, next) => {
    uploadSingle(req, res, (err) => { // call as a normal function
      if (err) {
        console.log('[TEST]',err)
        return res.status(400).send({success: false, message: "Only Images are allowed!"});;
      }

      const file = req.file;
      if (!file) {
        return res.status(400).send({success: false, message: "Please Upload A File!"});;
      }
      req['filename'] = filename;
      controller.makePredictions(req, res, next);
    })
  });
module.exports = router;