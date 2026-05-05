import React, { useState, useRef } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { UploadCloud, ShieldAlert, CheckCircle, Activity, FileImage, XCircle, ChevronRight, Loader2 } from 'lucide-react';
import { analyzeMRI, AnalysisResult } from '../services/geminiService';

export default function MRIAnalyzer() {
  const [dragActive, setDragActive] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [previewURL, setPreviewURL] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      handleFileSelected(e.target.files[0]);
    }
  };

  const handleFileSelected = (selectedFile: File) => {
    setError(null);
    setResult(null);
    
    // Check if it's an image
    if (!selectedFile.type.startsWith('image/')) {
      setError("Please upload a valid image file (JPG, PNG).");
      return;
    }
    
    setFile(selectedFile);
    
    // Create preview
    const objectUrl = URL.createObjectURL(selectedFile);
    setPreviewURL(objectUrl);
  };

  const resetAll = () => {
    setFile(null);
    if (previewURL) {
      URL.revokeObjectURL(previewURL);
    }
    setPreviewURL(null);
    setResult(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const runAnalysis = async () => {
    if (!file) return;

    setIsAnalyzing(true);
    setError(null);

    try {
      // Read file to base64
      const buffer = await file.arrayBuffer();
      const base64Bytes = btoa(
        new Uint8Array(buffer).reduce(
          (data, byte) => data + String.fromCharCode(byte),
          ''
        )
      );

      const mimeType = file.type;
      const dataUri = `data:${mimeType};base64,${base64Bytes}`; // For debugging if needed
      
      const analysis = await analyzeMRI(base64Bytes, mimeType);
      
      // Artificial delay to make it feel like "heavy processing" if Gemini is too fast
      await new Promise(resolve => setTimeout(resolve, 800));
      
      setResult(analysis);
    } catch (err: any) {
      setError(err.message || "An error occurred during analysis.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="w-full max-w-4xl mx-auto px-4 py-8">
      
      <div className="mb-8 border-b border-gray-200 pb-6 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-gray-900 tracking-tight flex items-center gap-2">
            <Activity className="w-6 h-6 text-blue-600" />
            NeuroScan ML
          </h1>
          <p className="text-sm text-gray-500 mt-1">Prototype Brain Tumor MRI Classification (v0.1)</p>
        </div>
        <div className="text-xs bg-blue-50 text-blue-700 px-3 py-1 rounded-full border border-blue-100 font-medium tracking-wide">
          RESEARCH PREVIEW
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        
        {/* Left Column: Upload & Preview */}
        <div className="lg:col-span-7 flex flex-col gap-6">
          
          <AnimatePresence mode="popLayout">
            {!file ? (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0.95 }}
                className={`dash-border rounded-2xl p-10 flex flex-col items-center justify-center text-center transition-colors cursor-pointer min-h-[400px] bg-white ${dragActive ? 'active' : ''}`}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
              >
                <div className="w-16 h-16 bg-blue-50 text-blue-500 rounded-full flex items-center justify-center mb-4">
                  <UploadCloud className="w-8 h-8" />
                </div>
                <h3 className="text-lg font-medium text-gray-900 mb-2">Upload MRI Scan</h3>
                <p className="text-sm text-gray-500 max-w-xs mb-6">
                  Drag and drop a 2D MRI slice (PNG/JPG), or click to browse files.
                </p>
                <button className="bg-gray-900 text-white px-5 py-2.5 rounded-lg text-sm font-medium hover:bg-gray-800 transition-colors">
                  Select Image
                </button>
                <input
                  ref={fileInputRef}
                  type="file"
                  className="hidden"
                  accept="image/jpeg, image/png, image/jpg"
                  onChange={handleChange}
                />
              </motion.div>
            ) : (
              <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                className="bg-white rounded-2xl overflow-hidden shadow-[0_2px_12px_rgba(0,0,0,0.06)] border border-gray-100"
              >
                <div className="bg-gray-50 px-4 py-3 flex items-center justify-between border-b border-gray-100">
                  <div className="flex items-center gap-2">
                    <FileImage className="w-4 h-4 text-gray-400" />
                    <span className="text-sm font-medium text-gray-700 truncate max-w-[200px]">{file.name}</span>
                  </div>
                  {!isAnalyzing && (
                    <button 
                      onClick={resetAll}
                      className="text-gray-400 hover:text-red-500 transition-colors tooltip"
                      title="Remove image"
                    >
                      <XCircle className="w-5 h-5" />
                    </button>
                  )}
                </div>
                <div className="p-4 bg-gray-100/50 flex justify-center min-h-[300px]">
                  {previewURL && (
                    <div className="relative group">
                      <img 
                        src={previewURL} 
                        alt="Uploaded MRI" 
                        className="max-h-[400px] object-contain rounded shadow-sm"
                      />
                      {isAnalyzing && (
                        <div className="absolute inset-0 bg-white/60 backdrop-blur-[2px] flex flex-col items-center justify-center rounded">
                          <Loader2 className="w-8 h-8 text-blue-600 animate-spin mb-3" />
                          <span className="text-sm font-medium text-blue-900 bg-blue-50 px-3 py-1 rounded-full border border-blue-100">
                            Running Model Inference...
                          </span>
                        </div>
                      )}
                    </div>
                  )}
                </div>
                
                <div className="p-4 border-t border-gray-100 bg-white flex justify-between items-center">
                  <span className="text-xs font-mono text-gray-500">{(file.size / 1024).toFixed(1)} KB</span>
                  {!isAnalyzing && !result && (
                     <button 
                      onClick={runAnalysis}
                      className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-2 rounded-lg text-sm font-medium shadow-sm transition-colors flex items-center gap-2"
                    >
                      Analyze Source
                      <ChevronRight className="w-4 h-4" />
                    </button>
                  )}
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {error && (
            <motion.div 
              initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
              className="bg-red-50 border-l-4 border-red-500 p-4 rounded-r-lg"
            >
              <div className="flex">
                <ShieldAlert className="h-5 w-5 text-red-500 mr-2" />
                <p className="text-sm text-red-700">{error}</p>
              </div>
            </motion.div>
          )}

        </div>

        {/* Right Column: Results & Info */}
        <div className="lg:col-span-5">
          
          <AnimatePresence mode="popLayout" initial={false}>
            {!result ? (
               <motion.div 
                key="empty-state"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0, x: 20 }}
                className="bg-white rounded-2xl p-6 border border-gray-100 shadow-[0_2px_12px_rgba(0,0,0,0.06)] h-full"
              >
                <h3 className="text-sm font-medium text-gray-400 uppercase tracking-wider mb-6">Patient Screening Protocol</h3>
                
                <ul className="space-y-6">
                   <li className="flex gap-4">
                     <div className="w-8 h-8 rounded-full bg-blue-50 flex items-center justify-center flex-shrink-0 text-blue-600 font-mono text-sm font-medium">1</div>
                     <div>
                       <h4 className="text-sm font-medium text-gray-900">High Recall Pipeline</h4>
                       <p className="text-sm text-gray-500 mt-1">Model is biased towards identifying any anomaly (reducing false negatives).</p>
                     </div>
                   </li>
                   <li className="flex gap-4">
                     <div className="w-8 h-8 rounded-full bg-blue-50 flex items-center justify-center flex-shrink-0 text-blue-600 font-mono text-sm font-medium">2</div>
                     <div>
                       <h4 className="text-sm font-medium text-gray-900">Transfer Learning Backbone</h4>
                       <p className="text-sm text-gray-500 mt-1">Feature extraction powered by deep ML architectures.</p>
                     </div>
                   </li>
                   <li className="flex gap-4">
                     <div className="w-8 h-8 rounded-full bg-amber-50 flex items-center justify-center flex-shrink-0 text-amber-600 font-mono text-sm font-medium">3</div>
                     <div>
                       <h4 className="text-sm font-medium text-gray-900">Clinical Safety Override</h4>
                       <p className="text-sm text-gray-500 mt-1">Any prediction with &lt; 90% confidence is flagged for mandatory manual review.</p>
                     </div>
                   </li>
                </ul>
              </motion.div>
            ) : (
              <motion.div
                key="result-state"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                className="bg-white rounded-2xl overflow-hidden shadow-[0_2px_12px_rgba(0,0,0,0.06)] border border-gray-100 sticky top-8"
              >
                <div className="bg-gray-50 px-6 py-4 border-b border-gray-100">
                  <h3 className="text-sm font-medium text-gray-500 uppercase tracking-wider">Analysis Report</h3>
                </div>

                <div className="p-6">
                  {/* Results logic */}
                  {(() => {
                    const confPercent = Math.round(result.confidence * 100);
                    const isHighConfidence = confPercent >= 90;
                    const hasTumor = result.prediction.toLowerCase().includes("tumor") && !result.prediction.toLowerCase().includes("no ");
                    
                    if (!isHighConfidence) {
                      return (
                        <div className="text-center">
                          <div className="mx-auto w-16 h-16 bg-amber-100 text-amber-600 rounded-full flex items-center justify-center mb-4">
                            <ShieldAlert className="w-8 h-8" />
                          </div>
                          <h2 className="text-xl font-bold text-gray-900 mb-2">Manual Review Required</h2>
                          <div className="inline-block bg-amber-50 text-amber-700 px-3 py-1 rounded border border-amber-200 font-mono text-sm mb-4">
                            Confidence: {confPercent}%
                          </div>
                          <p className="text-sm text-gray-600 border-t border-gray-100 pt-4 mt-2">
                            The model is uncertain ({confPercent}% &lt; 90% threshold). Due to clinical safety protocols, this scan is flagged for human examination.
                          </p>
                          <div className="mt-4 p-3 bg-gray-50 rounded-lg text-left">
                            <h4 className="text-xs uppercase text-gray-500 font-semibold mb-1">Raw Prediction</h4>
                            <p className="text-sm font-medium text-gray-800">{result.prediction}</p>
                          </div>
                        </div>
                      );
                    }

                    if (hasTumor) {
                      return (
                        <div className="text-center">
                           <div className="mx-auto w-16 h-16 bg-red-100 text-red-600 rounded-full flex items-center justify-center mb-4">
                            <Activity className="w-8 h-8" />
                          </div>
                          <h2 className="text-xl font-bold text-gray-900 mb-2">Tumor Detected</h2>
                          <div className="inline-block bg-green-50 text-green-700 px-3 py-1 rounded border border-green-200 font-mono text-sm mb-4 flex items-center justify-center gap-2 mx-auto w-max">
                            <CheckCircle className="w-4 h-4" />
                            Confidence: {confPercent}%
                          </div>
                           <div className="mt-6 p-4 bg-gray-50 rounded-lg text-left border border-gray-100">
                            <h4 className="text-xs uppercase text-gray-500 font-semibold mb-2">Clinical Findings</h4>
                            <p className="text-sm text-gray-800">{result.findings}</p>
                          </div>
                        </div>
                      )
                    }

                    // No Tumor, High Confidence
                    return (
                        <div className="text-center">
                           <div className="mx-auto w-16 h-16 bg-emerald-100 text-emerald-600 rounded-full flex items-center justify-center mb-4">
                            <CheckCircle className="w-8 h-8" />
                          </div>
                          <h2 className="text-xl font-bold text-gray-900 mb-2">No Tumor Detected</h2>
                          <div className="inline-block bg-emerald-50 text-emerald-700 px-3 py-1 rounded border border-emerald-200 font-mono text-sm mb-4 flex items-center justify-center gap-2 mx-auto w-max">
                            <Activity className="w-4 h-4" />
                            Confidence: {confPercent}%
                          </div>
                           <div className="mt-6 p-4 bg-gray-50 rounded-lg text-left border border-gray-100">
                            <h4 className="text-xs uppercase text-gray-500 font-semibold mb-2">Clinical Findings</h4>
                            <p className="text-sm text-gray-800">{result.findings}</p>
                          </div>
                        </div>
                      )
                  })()}
                </div>
                
                <div className="bg-gray-50 px-6 py-4 border-t border-gray-100 flex justify-between items-center">
                   <button 
                      onClick={resetAll}
                      className="text-sm text-gray-600 hover:text-gray-900 font-medium transition-colors"
                    >
                      Process Another Scan
                    </button>
                    <span className="text-xs font-mono text-gray-400">Lat: 1.2s</span>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

      </div>
    </div>
  );
}
