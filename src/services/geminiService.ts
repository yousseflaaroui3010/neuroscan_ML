import { GoogleGenAI } from '@google/genai';

// Lazily initialize to avoid crashing if the user hasn't set their key in the environment yet
let ai: GoogleGenAI | null = null;

export function getAI() {
  if (!ai) {
    const key = process.env.GEMINI_API_KEY;
    if (!key) {
      throw new Error("GEMINI_API_KEY environment variable is required.");
    }
    ai = new GoogleGenAI({ apiKey: key });
  }
  return ai;
}

export interface AnalysisResult {
  prediction: "Tumor Detected" | "No Tumor Detected";
  confidence: number;
  findings: string;
}

export async function analyzeMRI(base64Image: string, mimeType: string): Promise<AnalysisResult> {
  const ai = getAI();
  
  const prompt = `
    You are an AI acting as a prototype Brain Tumor classification model (acting as a proxy for an open-source PyTorch CNN).
    Analyze this single 2D slice from a Brain MRI scan.
    Determine whether there is a brain tumor or not.
    Be extremely meticulous. If you are uncertain or if the image quality is poor/ambiguous, provide a confidence score below 0.90.
    
    Respond ONLY in the exact JSON format below:
    {
      "prediction": "Tumor Detected" | "No Tumor Detected",
      "confidence": <number between 0.0 and 1.0>,
      "findings": "<Provide a brief, clinical 1-2 sentence description of what you observe.>"
    }
  `;

  // We remove the data:image/png;base64, prefix if it exists
  const cleanBase64 = base64Image.split(',')[1] || base64Image;

  const response = await ai.models.generateContent({
    model: 'gemini-2.5-flash',
    contents: [
      {
        role: 'user',
        parts: [
          {
            inlineData: {
              data: cleanBase64,
              mimeType: mimeType,
            },
          },
          { text: prompt },
        ],
      },
    ],
    config: {
      responseMimeType: "application/json",
      temperature: 0.1,
    }
  });

  const text = response.text || "{}";
  try {
    const data = JSON.parse(text) as AnalysisResult;
    return {
      prediction: data.prediction || "No Tumor Detected",
      confidence: data.confidence || 0,
      findings: data.findings || "Image analysis completed with missing findings output.",
    };
  } catch (error) {
    console.error("Failed to parse Gemini response:", text);
    throw new Error("Failed to parse the analysis result.");
  }
}
