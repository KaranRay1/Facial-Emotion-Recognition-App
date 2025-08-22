import React from 'react';

interface EmotionData {
  emotion: string;
  confidence: number;
  color: string;
}

// Real API call to backend Flask server
export const analyzeEmotion = async (imageData: string): Promise<{emotions: EmotionData[], age: number | null, gender: string | null}> => {
  const colorMap: Record<string, string> = {
    happy: '#10B981',
    sad: '#3B82F6',
    angry: '#EF4444',
    surprised: '#F59E0B',
    neutral: '#6B7280',
    fear: '#8B5CF6',
    disgust: '#EC4899',
  };
  try {
    const response = await fetch('http://localhost:5000/analyze-emotion', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: imageData }),
    });
    const result = await response.json();
    if (result.error) throw new Error(result.error);
    const emotionObj = result.emotion || {};
    const emotions: EmotionData[] = Object.entries(emotionObj).map(([emotion, confidence]) => ({
      emotion,
      confidence: typeof confidence === 'number' ? confidence : 0,
      color: colorMap[emotion] || '#6B7280',
    }));
    return {
      emotions: emotions.sort((a, b) => b.confidence - a.confidence),
      age: typeof result.age === 'number' ? result.age : null,
      gender: typeof result.gender === 'string' ? result.gender : null,
    };
  } catch (err) {
    return { emotions: [], age: null, gender: null };
  }
};

// Integration guide component for developers
const ModelIntegration: React.FC = () => {
  return (
    <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl p-6 border border-blue-200">
      <h3 className="text-lg font-semibold text-blue-800 mb-3">
        🔗 Model Integration Guide
      </h3>
      <div className="space-y-3 text-sm text-blue-700">
        <p>
          <strong>Replace the mock function</strong> in <code className="bg-blue-100 px-2 py-1 rounded">analyzeEmotion</code> with your actual deep learning model integration:
        </p>
        <ul className="list-disc pl-5 space-y-1">
          <li>TensorFlow.js models: Load and run inference directly in the browser</li>
          <li>API endpoints: Send image data to your Python/Flask/FastAPI backend</li>
          <li>Cloud services: Integrate with AWS Rekognition, Google Cloud Vision, or Azure Face API</li>
          <li>ONNX models: Use ONNX.js for optimized model inference</li>
        </ul>
        <p className="mt-3">
          The function should return an array of emotion objects with confidence scores between 0-1.
        </p>
      </div>
    </div>
  );
};

export default ModelIntegration;