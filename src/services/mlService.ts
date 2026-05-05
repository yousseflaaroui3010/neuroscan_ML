import * as tf from '@tensorflow/tfjs';

let featureExtractor: tf.GraphModel | null = null;
let classifier: tf.Sequential | null = null;

// Load the pre-trained MobileNet feature extractor
export const loadFeatureExtractor = async () => {
  if (!featureExtractor) {
    // Loads MobileNet V3 small feature vector model
    const MODEL_URL = 'https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v3_small_100_224/feature_vector/5/default/1';
    featureExtractor = await tf.loadGraphModel(MODEL_URL, { fromTFHub: true });
  }
  return featureExtractor;
};

// Preprocess image and extract features
export const extractFeatures = async (imageElement: HTMLImageElement | HTMLCanvasElement): Promise<tf.Tensor> => {
  return tf.tidy(() => {
    // 1. Read pixels
    let img = tf.browser.fromPixels(imageElement);
    // 2. Resize to 224x224 (expected by MobileNet)
    img = tf.image.resizeBilinear(img, [224, 224]);
    // 3. Normalize to [0, 1] range
    img = img.div(255.0).expandDims(0);
    // 4. Extract features
    if (!featureExtractor) throw new Error("Feature extractor not loaded");
    return featureExtractor.predict(img) as tf.Tensor;
  });
};

export interface LabeledSample {
  image: HTMLImageElement;
  label: number; // 0 for No Tumor, 1 for Tumor
}

// Train a new classification head on top of the extracted features
export const trainClassifier = async (
  samples: LabeledSample[],
  onEpochEnd: (epoch: number, logs: tf.Logs) => void
) => {
  if (samples.length === 0) throw new Error("No data provided");

  // Load extractor if not already loaded
  await loadFeatureExtractor();

  // 1. Extract features for all samples
  const xsArray: tf.Tensor[] = [];
  const ysArray: number[] = [];

  for (const sample of samples) {
    const features = await extractFeatures(sample.image);
    xsArray.push(features);
    ysArray.push(sample.label);
  }

  // 2. Concatenate tensors into batches
  const xs = tf.concat(xsArray, 0);
  const ys = tf.tensor1d(ysArray, 'int32');
  const ysOneHot = tf.oneHot(ys, 2); // 2 classes: No Tumor, Tumor

  // 3. Build Neural Network Classifier
  classifier = tf.sequential({
    layers: [
      tf.layers.dense({ units: 128, activation: 'relu', inputShape: [xs.shape[1] as number] }),
      tf.layers.dropout({ rate: 0.3 }),
      tf.layers.dense({ units: 64, activation: 'relu' }),
      tf.layers.dense({ units: 2, activation: 'softmax' })
    ]
  });

  classifier.compile({
    optimizer: tf.train.adam(0.001),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });

  // 4. Train the model
  const history = await classifier.fit(xs, ysOneHot, {
    epochs: 25,
    batchSize: Math.min(32, xs.shape[0]),
    validationSplit: 0.2, // Use 20% of data for validation
    shuffle: true,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        if (logs) onEpochEnd(epoch, logs);
      }
    }
  });

  // 5. Cleanup memory
  xs.dispose();
  ys.dispose();
  ysOneHot.dispose();
  for (const t of xsArray) t.dispose();

  return history;
};

// Evaluate the model to get Accuracy, Precision, Recall, Confusion Matrix
export const evaluateModel = async (samples: LabeledSample[]) => {
  if (!classifier) throw new Error("Model is not trained yet.");

  let tp = 0, tn = 0, fp = 0, fn = 0;

  for (const sample of samples) {
    const features = await extractFeatures(sample.image);
    const predictionTensor = classifier.predict(features) as tf.Tensor;
    const predictionData = await predictionTensor.data();
    
    // Class 0 = No Tumor, Class 1 = Tumor
    const predictedClass = predictionData[1] > predictionData[0] ? 1 : 0;
    
    const actual = sample.label;
    
    if (actual === 1 && predictedClass === 1) tp++;
    if (actual === 0 && predictedClass === 0) tn++;
    if (actual === 0 && predictedClass === 1) fp++;
    if (actual === 1 && predictedClass === 0) fn++;
    
    features.dispose();
    predictionTensor.dispose();
  }

  const accuracy = (tp + tn) / samples.length;
  const precision = tp / (tp + fp || 1); // Avoid division by zero
  const recall = tp / (tp + fn || 1);

  return {
    accuracy,
    precision,
    recall,
    confusionMatrix: [
      [tn, fp], // [True Neg, False Pos]
      [fn, tp]  // [False Neg, True Pos]
    ]
  };
};

// Return prediction and confidence for a single image
export const predictImage = async (imageElement: HTMLImageElement) => {
  if (!classifier) throw new Error("Model is not trained yet.");
  await loadFeatureExtractor();
  
  const features = await extractFeatures(imageElement);
  const predictionTensor = classifier.predict(features) as tf.Tensor;
  const predictionData = await predictionTensor.data();
  
  const tumorProbability = predictionData[1];
  const noTumorProbability = predictionData[0];
  
  features.dispose();
  predictionTensor.dispose();
  
  return {
    tumorProbability, // Confidence of Tumor
    noTumorProbability // Confidence of No Tumor
  };
};

export const isModelTrained = () => {
    return classifier !== null;
};
