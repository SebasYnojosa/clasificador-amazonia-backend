import express, { Request, Response } from 'express';
import multer from 'multer';
import * as tf from '@tensorflow/tfjs';
import fs from 'fs';
import { createCanvas, loadImage } from 'canvas';

const app = express();
const port = 3000;

// Configurar multer para manejar la carga de archivos
const upload = multer({ dest: 'uploads/' });

app.post('/predict', upload.single('image'), async (req: Request, res: Response): Promise<void> => {
  if (!req.file) {
    res.status(400).send('No file uploaded.');
    return;
  }

  try {
    // Cargar la imagen
    const imageBuffer = fs.readFileSync(req.file.path);
    const img = await loadImage(imageBuffer);
    const canvas = createCanvas(img.width, img.height);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0);
    
    // Obtener los datos de la imagen del canvas
    const imageData = ctx.getImageData(0, 0, img.width, img.height);
    const { data, width, height } = imageData;

    // Crear un tensor a partir de los datos de la imagen
    const imageTensor = tf.tensor3d(new Uint8Array(data.buffer), [height, width, 4]);

    // Cargar el modelo
    const model = await tf.loadLayersModel('file://path/to/your/model/model.json');

    // Preprocesar la imagen y hacer la predicción
    const resizedImage = tf.image.resizeBilinear(imageTensor, [224, 224]);
    const normalizedImage = resizedImage.div(255.0).expandDims(0);
    const prediction = model.predict(normalizedImage) as tf.Tensor;

    // Enviar la predicción como respuesta
    const predictionArray = prediction.arraySync();
    res.json({ prediction: predictionArray });

    // Eliminar el archivo temporal
    fs.unlinkSync(req.file.path);
  } catch (error) {
    console.error('Error processing image:', error);
    res.status(500).send('Error processing image.');
  }
});

app.listen(port, () => {
  console.log(`Server is running on http://localhost:${port}`);
});