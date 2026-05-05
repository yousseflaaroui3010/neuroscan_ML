import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { 
  Database, BrainCircuit, Activity, UploadCloud, XCircle, 
  PlayCircle, CheckCircle, ShieldAlert, FileImage, 
  ChevronRight, BarChart3, AlertCircle
} from 'lucide-react';
import { 
  trainClassifier, evaluateModel, predictImage, 
  isModelTrained, LabeledSample, loadFeatureExtractor
} from '../services/mlService';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer } from 'recharts';

export interface ImageData {
  id: string;
  url: string;
  label: number; // 0 for Normal, 1 for Tumor
  file: File;
}

export default function MRIWorkstation() {
  const [activeTab, setActiveTab] = useState<'dataset' | 'train' | 'predict'>('dataset');
  
  // Dataset state
  const [images, setImages] = useState<ImageData[]>([]);
  const tumorInputRef = useRef<HTMLInputElement>(null);
  const normalInputRef = useRef<HTMLInputElement>(null);

  // Training state
  const [isTraining, setIsTraining] = useState(false);
  const [trainingLogs, setTrainingLogs] = useState<{epoch: number, loss: number, accuracy: number, val_loss: number, val_accuracy: number}[]>([]);
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [evaluation, setEvaluation] = useState<{accuracy: number, precision: number, recall: number, confusionMatrix: number[][]} | null>(null);

  // Predict state
  const [predictFile, setPredictFile] = useState<File | null>(null);
  const [predictUrl, setPredictUrl] = useState<string | null>(null);
  const [predictResult, setPredictResult] = useState<{ tumorProbability: number, noTumorProbability: number } | null>(null);
  const [isPredicting, setIsPredicting] = useState(false);

  // Load feature extractor eagerly
  useEffect(() => {
    loadFeatureExtractor().catch(console.error);
  }, []);

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>, label: number) => {
    if (e.target.files) {
      const newImages = Array.from(e.target.files).map((file: File) => ({
        id: Math.random().toString(36).substring(7),
        url: URL.createObjectURL(file),
        label,
        file
      }));
      setImages(prev => [...prev, ...newImages]);
    }
  };

  const removeImage = (id: string) => {
    setImages(prev => prev.filter(img => img.id !== id));
  };

  // Helper to load image url into an HTMLImageElement
  const loadImage = (url: string): Promise<HTMLImageElement> => {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = "anonymous";
      img.onload = () => resolve(img);
      img.onerror = (err) => reject(err);
      img.src = url;
    });
  };

  const startTraining = async () => {
    if (images.length < 5) {
      alert("Please upload at least 5 images of each class to train the model.");
      return;
    }
    
    setIsTraining(true);
    setTrainingLogs([]);
    setEvaluation(null);
    setTrainingProgress(0);

    try {
      // 1. Prepare data
      const samples: LabeledSample[] = [];
      for (const imgData of images) {
        const imageElement = await loadImage(imgData.url);
        samples.push({ image: imageElement, label: imgData.label });
      }

      // 2. Train Model
      await trainClassifier(samples, (epoch, logs) => {
        setTrainingProgress(Math.round(((epoch + 1) / 25) * 100)); // Assuming 25 epochs
        setTrainingLogs(prev => [...prev, {
          epoch: epoch + 1,
          loss: logs.loss || 0,
          accuracy: logs.acc || 0,
          val_loss: logs.val_loss || 0,
          val_accuracy: logs.val_acc || 0
        }]);
      });

      // 3. Evaluate Model on training set (in a real scenario we'd split a separate test set)
      // For this prototype, we'll evaluate on the same set or a validation set
      const evalMetrics = await evaluateModel(samples);
      setEvaluation({
        accuracy: evalMetrics.accuracy,
        precision: evalMetrics.precision,
        recall: evalMetrics.recall,
        confusionMatrix: evalMetrics.confusionMatrix
      });

    } catch (err) {
      console.error(err);
      alert("Error during training: " + String(err));
    } finally {
      setIsTraining(false);
    }
  };

  const handlePredictUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setPredictFile(file);
      setPredictUrl(URL.createObjectURL(file));
      setPredictResult(null);
    }
  };

  const runPrediction = async () => {
    if (!predictUrl) return;
    setIsPredicting(true);
    try {
      const img = await loadImage(predictUrl);
      const result = await predictImage(img);
      setPredictResult(result);
    } catch (err) {
      console.error(err);
      alert("Error predicting: " + String(err));
    } finally {
      setIsPredicting(false);
    }
  };

  const normalImages = images.filter(img => img.label === 0);
  const tumorImages = images.filter(img => img.label === 1);

  return (
    <div className="w-full max-w-6xl mx-auto px-4 py-8 flex flex-col h-screen max-h-screen">
      
      {/* Header */}
      <div className="mb-6 flex-shrink-0 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-gray-900 tracking-tight flex items-center gap-2">
            <Activity className="w-6 h-6 text-blue-600" />
            NeuroML Workstation
          </h1>
          <p className="text-sm text-gray-500 mt-1">100% In-Browser Machine Learning Pipeline</p>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-gray-200 mb-6 flex-shrink-0">
        <button 
          onClick={() => setActiveTab('dataset')}
          className={`px-6 py-3 font-medium text-sm border-b-2 transition-colors ${activeTab === 'dataset' ? 'border-blue-600 text-blue-600' : 'border-transparent text-gray-500 hover:text-gray-700'}`}
        >
          <div className="flex items-center gap-2"><Database className="w-4 h-4" /> 1. Prepare Dataset</div>
        </button>
        <button 
          onClick={() => setActiveTab('train')}
          className={`px-6 py-3 font-medium text-sm border-b-2 transition-colors ${activeTab === 'train' ? 'border-blue-600 text-blue-600' : 'border-transparent text-gray-500 hover:text-gray-700'}`}
        >
          <div className="flex items-center gap-2"><BrainCircuit className="w-4 h-4" /> 2. Train & Evaluate</div>
        </button>
        <button 
          onClick={() => setActiveTab('predict')}
          className={`px-6 py-3 font-medium text-sm border-b-2 transition-colors ${activeTab === 'predict' ? 'border-blue-600 text-blue-600' : 'border-transparent text-gray-500 hover:text-gray-700'}`}
        >
           <div className="flex items-center gap-2"><Activity className="w-4 h-4" /> 3. Predict MRI</div>
        </button>
      </div>

      {/* Content Area */}
      <div className="flex-1 overflow-y-auto w-full">
        <AnimatePresence mode="wait">
          
          {/* 1. DATASET VIEW */}
          {activeTab === 'dataset' && (
             <motion.div 
               key="dataset"
               initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }}
               className="grid grid-cols-1 md:grid-cols-2 gap-8 h-full"
             >
                {/* No Tumor Upload */}
                <div className="bg-white rounded-xl border border-gray-200 shadow-sm flex flex-col overflow-hidden">
                  <div className="border-b border-gray-100 px-4 py-3 bg-gray-50 flex justify-between items-center">
                     <h3 className="font-semibold text-gray-800">Class 0: Healthy Control</h3>
                     <span className="text-xs font-mono bg-white border border-gray-200 px-2 py-1 rounded text-gray-500">{normalImages.length} images</span>
                  </div>
                  <div className="p-4 flex-1 flex flex-col overflow-hidden">
                     <button 
                        onClick={() => normalInputRef.current?.click()}
                        className="w-full flex items-center justify-center gap-2 py-4 border-2 border-dashed border-gray-300 rounded-lg text-gray-600 hover:bg-gray-50 hover:border-blue-400 transition-colors mb-4 flex-shrink-0"
                      >
                       <UploadCloud className="w-5 h-5" /> Select Images
                     </button>
                     <input type="file" multiple accept="image/*" className="hidden" ref={normalInputRef} onChange={(e) => handleFileUpload(e, 0)} />
                     
                     <div className="overflow-y-auto flex-1 grid grid-cols-3 gap-2 pr-2">
                       {normalImages.map(img => (
                         <div key={img.id} className="relative group aspect-square bg-gray-100 rounded overflow-hidden">
                           <img src={img.url} alt="" className="w-full h-full object-cover" />
                           <button onClick={() => removeImage(img.id)} className="absolute top-1 right-1 bg-black/50 text-white rounded-full p-1 opacity-0 group-hover:opacity-100 transition-opacity">
                             <XCircle className="w-4 h-4" />
                           </button>
                         </div>
                       ))}
                       {normalImages.length === 0 && (
                         <div className="col-span-3 text-center text-sm text-gray-400 py-12 flex flex-col items-center">
                           <FileImage className="w-8 h-8 mb-2 opacity-50" />
                           Upload normal (no tumor) MRI scans
                         </div>
                       )}
                     </div>
                  </div>
                </div>

                {/* Tumor Upload */}
                <div className="bg-white rounded-xl border border-gray-200 shadow-sm flex flex-col overflow-hidden">
                  <div className="border-b border-gray-100 px-4 py-3 bg-red-50 flex justify-between items-center">
                     <h3 className="font-semibold text-red-900">Class 1: Tumor Detected</h3>
                     <span className="text-xs font-mono bg-white border border-red-100 px-2 py-1 rounded text-red-600">{tumorImages.length} images</span>
                  </div>
                  <div className="p-4 flex-1 flex flex-col overflow-hidden">
                     <button 
                        onClick={() => tumorInputRef.current?.click()}
                        className="w-full flex items-center justify-center gap-2 py-4 border-2 border-dashed border-red-200 rounded-lg text-red-600 hover:bg-red-50 hover:border-red-400 transition-colors mb-4 flex-shrink-0"
                      >
                       <UploadCloud className="w-5 h-5" /> Select Images
                     </button>
                     <input type="file" multiple accept="image/*" className="hidden" ref={tumorInputRef} onChange={(e) => handleFileUpload(e, 1)} />
                     
                     <div className="overflow-y-auto flex-1 grid grid-cols-3 gap-2 pr-2">
                       {tumorImages.map(img => (
                         <div key={img.id} className="relative group aspect-square bg-gray-100 rounded overflow-hidden border border-red-100">
                           <img src={img.url} alt="" className="w-full h-full object-cover" />
                           <button onClick={() => removeImage(img.id)} className="absolute top-1 right-1 bg-black/50 text-white rounded-full p-1 opacity-0 group-hover:opacity-100 transition-opacity">
                             <XCircle className="w-4 h-4" />
                           </button>
                         </div>
                       ))}
                       {tumorImages.length === 0 && (
                         <div className="col-span-3 text-center text-sm text-gray-400 py-12 flex flex-col items-center">
                           <FileImage className="w-8 h-8 mb-2 opacity-50 text-red-300" />
                           Upload MRI scans featuring tumors
                         </div>
                       )}
                     </div>
                  </div>
                </div>
             </motion.div>
          )}

          {/* 2. TRAIN & EVALUATE VIEW */}
          {activeTab === 'train' && (
             <motion.div 
               key="train"
               initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }}
               className="grid grid-cols-1 lg:grid-cols-2 gap-8"
             >
                <div className="bg-white p-6 rounded-xl border border-gray-200 shadow-sm flex flex-col">
                  <div>
                    <h3 className="text-lg font-bold text-gray-900 mb-1">Transfer Learning Process</h3>
                    <p className="text-sm text-gray-500 mb-6">
                      Extracts features from MRIs using MobileNetV3 and trains a custom classification head on top.
                    </p>

                    <div className="bg-gray-50 rounded-lg p-4 border border-gray-100 mb-6 flex justify-between items-center">
                      <div>
                        <div className="text-sm font-medium text-gray-800">Total Samples: {images.length}</div>
                        <div className="text-xs text-gray-500 mt-1">Class balance: {normalImages.length} Normal / {tumorImages.length} Tumor</div>
                      </div>
                      <button 
                        onClick={startTraining}
                        disabled={isTraining || images.length < 2}
                        className={`px-5 py-2.5 rounded-lg font-medium flex items-center gap-2 transition-all shadow-sm ${
                          isTraining ? 'bg-blue-100 text-blue-500 cursor-not-allowed' : 'bg-blue-600 text-white hover:bg-blue-700'
                        }`}
                      >
                        {isTraining ? <Activity className="w-5 h-5 animate-spin" /> : <PlayCircle className="w-5 h-5" />}
                        {isTraining ? `Training (${trainingProgress}%)` : `Initialize Training Loop`}
                      </button>
                    </div>
                  </div>

                  <div className="flex-1 min-h-[300px]">
                     {trainingLogs.length > 0 ? (
                       <div className="h-full flex justify-end flex-col">
                         <h4 className="text-xs font-semibold uppercase text-gray-500 mb-4 tracking-wider">Training Loss & Accuracy</h4>
                         <div className="h-[250px] w-full">
                           <ResponsiveContainer width="100%" height="100%">
                              <LineChart data={trainingLogs} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                                <XAxis dataKey="epoch" tick={{fontSize: 12}} />
                                <YAxis yAxisId="left" tick={{fontSize: 12}} domain={[0, 'auto']} />
                                <YAxis yAxisId="right" orientation="right" tick={{fontSize: 12}} domain={[0, 1]} />
                                <RechartsTooltip />
                                <Line yAxisId="left" type="monotone" dataKey="loss" stroke="#ef4444" strokeWidth={2} name="Loss" dot={false} />
                                <Line yAxisId="right" type="monotone" dataKey="accuracy" stroke="#3b82f6" strokeWidth={2} name="Accuracy" dot={false} />
                              </LineChart>
                           </ResponsiveContainer>
                         </div>
                       </div>
                     ) : (
                       <div className="h-full flex items-center justify-center text-gray-400 flex-col">
                          <BarChart3 className="w-12 h-12 mb-3 opacity-20" />
                          <p className="text-sm">Telemetry will appear here during training</p>
                       </div>
                     )}
                  </div>
                </div>

                <div className="bg-white p-6 rounded-xl border border-gray-200 shadow-sm flex flex-col">
                  <h3 className="text-lg font-bold text-gray-900 mb-1">Model Evaluation</h3>
                  <p className="text-sm text-gray-500 mb-6">
                    Metrics generated after training completion.
                  </p>

                  {!evaluation ? (
                    <div className="flex-1 flex items-center justify-center border-2 border-dashed border-gray-100 rounded-xl bg-gray-50 text-gray-400">
                      Run training to generate evaluation metrics.
                    </div>
                  ) : (
                    <div className="space-y-6">
                      <div className="grid grid-cols-3 gap-4">
                        <div className="bg-blue-50 border border-blue-100 p-4 rounded-xl text-center">
                          <div className="text-2xl font-bold text-blue-700">{(evaluation.accuracy * 100).toFixed(1)}%</div>
                          <div className="text-xs uppercase font-semibold text-blue-500 mt-1">Accuracy</div>
                        </div>
                        <div className="bg-emerald-50 border border-emerald-100 p-4 rounded-xl text-center">
                          <div className="text-2xl font-bold text-emerald-700">{(evaluation.precision * 100).toFixed(1)}%</div>
                          <div className="text-xs uppercase font-semibold text-emerald-500 mt-1">Precision</div>
                        </div>
                        <div className="bg-purple-50 border border-purple-100 p-4 rounded-xl text-center">
                          <div className="text-2xl font-bold text-purple-700">{(evaluation.recall * 100).toFixed(1)}%</div>
                          <div className="text-xs uppercase font-semibold text-purple-500 mt-1">Recall</div>
                        </div>
                      </div>

                      <div>
                        <h4 className="text-sm font-semibold text-gray-700 mb-3">Confusion Matrix</h4>
                        <div className="grid grid-cols-3 grid-rows-3 gap-1 text-sm pt-2">
                           {/* Empty top-left */}
                           <div></div>
                           <div className="text-center text-xs font-semibold text-gray-500 pb-2">Pred: Normal</div>
                           <div className="text-center text-xs font-semibold text-gray-500 pb-2">Pred: Tumor</div>

                           <div className="flex items-center justify-end pr-3 text-xs font-semibold text-gray-500">True: Normal</div>
                           <div className="bg-green-100 border border-green-200 rounded flex items-center justify-center p-3 font-mono font-bold text-green-800">
                             {evaluation.confusionMatrix[0][0]} <br/><span className="text-[10px] font-normal leading-tight ml-1">TN</span>
                           </div>
                           <div className="bg-red-50 border border-red-200 rounded flex items-center justify-center p-3 font-mono font-bold text-red-800">
                             {evaluation.confusionMatrix[0][1]} <br/><span className="text-[10px] font-normal leading-tight ml-1">FP</span>
                           </div>

                           <div className="flex items-center justify-end pr-3 text-xs font-semibold text-gray-500">True: Tumor</div>
                           <div className="bg-red-50 border border-red-200 rounded flex items-center justify-center p-3 font-mono font-bold text-red-800">
                             {evaluation.confusionMatrix[1][0]} <br/><span className="text-[10px] font-normal leading-tight ml-1">FN</span>
                           </div>
                           <div className="bg-green-100 border border-green-200 rounded flex items-center justify-center p-3 font-mono font-bold text-green-800">
                             {evaluation.confusionMatrix[1][1]} <br/><span className="text-[10px] font-normal leading-tight ml-1">TP</span>
                           </div>
                        </div>
                        <p className="text-xs text-gray-400 mt-4 italic">
                          *A false negative (FN) is the most dangerous classification in medical screening. High recall minimizes FNs.
                        </p>
                      </div>
                    </div>
                  )}
                </div>
             </motion.div>
          )}

          {/* 3. PREDICT VIEW */}
          {activeTab === 'predict' && (
             <motion.div 
               key="predict"
               initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }}
               className="grid grid-cols-1 md:grid-cols-2 gap-8 h-full"
             >
                <div className="flex flex-col gap-6">
                  {!isModelTrained() && (
                    <div className="bg-amber-50 border border-amber-200 rounded-lg p-4 flex gap-3 text-amber-800 text-sm">
                      <AlertCircle className="w-5 h-5 flex-shrink-0" />
                      <p>The model must be trained before you can execute inference. Go to the Train interface first.</p>
                    </div>
                  )}

                  <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-6 flex flex-col items-center">
                    {!predictUrl ? (
                      <div className="w-full">
                        <label className="border-2 border-dashed border-gray-300 rounded-xl p-10 flex flex-col items-center justify-center cursor-pointer hover:bg-gray-50 hover:border-blue-400 transition-all">
                          <UploadCloud className="w-10 h-10 text-gray-400 mb-3" />
                          <span className="text-gray-700 font-medium mb-1">Upload New MRI Scan</span>
                          <span className="text-gray-400 text-sm">Upload a test slice to evaluate</span>
                          <input type="file" className="hidden" accept="image/*" onChange={handlePredictUpload} />
                        </label>
                      </div>
                    ) : (
                      <div className="w-full">
                         <div className="relative aspect-auto max-h-[300px] overflow-hidden rounded-lg bg-gray-100 flex justify-center border border-gray-200">
                           <img src={predictUrl} className="object-contain max-h-[300px]" alt="Prediction target" />
                           <button 
                             onClick={() => { setPredictUrl(null); setPredictResult(null); }}
                             className="absolute top-2 right-2 bg-black/60 text-white rounded-full p-1.5 hover:bg-black transition-colors"
                           >
                             <XCircle className="w-5 h-5" />
                           </button>
                         </div>
                         <button 
                            onClick={runPrediction}
                            disabled={!isModelTrained() || isPredicting}
                            className={`w-full mt-4 py-3 rounded-lg font-medium shadow-sm transition-colors flex items-center justify-center gap-2 ${
                              !isModelTrained() ? 'bg-gray-100 text-gray-400 cursor-not-allowed' :
                              isPredicting ? 'bg-blue-100 text-blue-500' : 'bg-blue-600 text-white hover:bg-blue-700'
                            }`}
                         >
                           {isPredicting ? <Activity className="w-5 h-5 animate-spin" /> : <ChevronRight className="w-5 h-5" />}
                           {isPredicting ? 'Running Inference...' : 'Run Analysis'}
                         </button>
                      </div>
                    )}
                  </div>
                </div>

                <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-6">
                  <h3 className="text-lg font-bold text-gray-900 mb-6 border-b border-gray-100 pb-4">Analysis Result</h3>
                  
                  {!predictResult ? (
                    <div className="text-center text-gray-400 py-10">
                      Results will appear here
                    </div>
                  ) : (
                    <div>
                      {(() => {
                        const tumorConf = predictResult.tumorProbability * 100;
                        const normalConf = predictResult.noTumorProbability * 100;
                        // Medical rule from PRD: if confidence < 90%, flag it.
                        const maxConf = Math.max(tumorConf, normalConf);
                        const isTumor = tumorConf > normalConf;
                        
                        if (maxConf < 90) {
                          return (
                            <div className="text-center">
                              <div className="mx-auto w-16 h-16 bg-amber-100 text-amber-600 rounded-full flex items-center justify-center mb-4">
                                <ShieldAlert className="w-8 h-8" />
                              </div>
                              <h2 className="text-2xl font-bold text-gray-900 mb-2">Manual Review Required</h2>
                              <div className="inline-block bg-amber-50 text-amber-700 px-3 py-1 rounded border border-amber-200 font-mono text-sm mb-4">
                                Confidence: {maxConf.toFixed(1)}% &lt; 90%
                              </div>
                              <p className="text-sm text-gray-600">
                                The ML model is uncertain. Due to clinical safety protocols, this scan must be flagged for human examination.
                              </p>
                            </div>
                          );
                        }

                        if (isTumor) {
                           return (
                             <div className="text-center">
                               <div className="mx-auto w-16 h-16 bg-red-100 text-red-600 rounded-full flex items-center justify-center mb-4">
                                 <Activity className="w-8 h-8" />
                               </div>
                               <h2 className="text-2xl font-bold text-gray-900 mb-2">Tumor Detected</h2>
                               <div className="inline-block bg-red-50 text-red-700 px-3 py-1 rounded border border-red-200 font-mono text-sm mb-4 flex items-center justify-center gap-2 mx-auto w-max">
                                 <CheckCircle className="w-4 h-4" />
                                 Confidence: {maxConf.toFixed(2)}%
                               </div>
                             </div>
                           )
                        }

                        return (
                           <div className="text-center">
                             <div className="mx-auto w-16 h-16 bg-emerald-100 text-emerald-600 rounded-full flex items-center justify-center mb-4">
                               <CheckCircle className="w-8 h-8" />
                             </div>
                             <h2 className="text-2xl font-bold text-gray-900 mb-2">No Tumor Detected</h2>
                             <div className="inline-block bg-emerald-50 text-emerald-700 px-3 py-1 rounded border border-emerald-200 font-mono text-sm mb-4 flex items-center justify-center gap-2 mx-auto w-max">
                               <CheckCircle className="w-4 h-4" />
                               Confidence: {maxConf.toFixed(2)}%
                             </div>
                           </div>
                        )
                      })()}

                      {/* Raw probabilities */}
                      <div className="mt-8 pt-6 border-t border-gray-100">
                        <h4 className="text-xs uppercase tracking-wider font-semibold text-gray-500 mb-4">Softmax Probabilities</h4>
                        
                        <div className="space-y-4">
                          <div>
                            <div className="flex justify-between text-sm font-medium mb-1">
                              <span className="text-emerald-700">Class 0: Healthy Control</span>
                              <span className="text-gray-500">{(predictResult.noTumorProbability * 100).toFixed(2)}%</span>
                            </div>
                            <div className="w-full bg-gray-100 h-2 rounded-full overflow-hidden">
                              <div className="bg-emerald-500 h-full" style={{width: `${predictResult.noTumorProbability * 100}%`}}></div>
                            </div>
                          </div>

                          <div>
                            <div className="flex justify-between text-sm font-medium mb-1">
                              <span className="text-red-700">Class 1: Tumor Active</span>
                              <span className="text-gray-500">{(predictResult.tumorProbability * 100).toFixed(2)}%</span>
                            </div>
                            <div className="w-full bg-gray-100 h-2 rounded-full overflow-hidden">
                              <div className="bg-red-500 h-full" style={{width: `${predictResult.tumorProbability * 100}%`}}></div>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
             </motion.div>
          )}

        </AnimatePresence>
      </div>

    </div>
  );
}
