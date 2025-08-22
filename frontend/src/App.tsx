import React, { useState } from 'react';
import { Brain, Activity } from 'lucide-react';
import CameraCapture from './components/CameraCapture';
import ImageUpload from './components/ImageUpload';
import EmotionChart from './components/EmotionChart';
import ModelIntegration, { analyzeEmotion } from './components/ModelIntegration';
import SmartDeviceIntegration from './components/SmartDeviceIntegration';
import BiasReporting from './components/BiasReporting';
import AvatarMirror from './components/AvatarMirror';

interface EmotionData {
  emotion: string;
  confidence: number;
  color: string;
}

function App() {
  const [emotions, setEmotions] = useState<EmotionData[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisCount, setAnalysisCount] = useState(0);
  const [age, setAge] = useState<number | null>(null);
  const [gender, setGender] = useState<string | null>(null);

  const handleImageAnalysis = async (imageData: string) => {
    setIsAnalyzing(true);
    try {
      const results = await analyzeEmotion(imageData);
      setEmotions(results.emotions);
      setAge(results.age);
      setGender(results.gender);
      setAnalysisCount(prev => prev + 1);
    } catch (error) {
      console.error('Error analyzing emotion:', error);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const topEmotion = emotions.length > 0 ? emotions[0] : null;

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100">
      {/* Header */}
      <div className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-3">
              <div className="bg-gradient-to-r from-blue-500 to-indigo-600 p-2 rounded-lg">
                <Brain className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-gray-900">
                  Emotion Recognition System
                </h1>
                <p className="text-sm text-gray-600">
                  AI-powered facial emotion analysis
                </p>
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2 bg-green-50 px-3 py-1 rounded-full">
                <Activity className="w-4 h-4 text-green-600" />
                <span className="text-sm font-medium text-green-800">
                  {analysisCount} analyses completed
                </span>
              </div>
              
              {topEmotion && (
                <div className="flex items-center space-x-2 bg-blue-50 px-3 py-1 rounded-full">
                  <div 
                    className="w-3 h-3 rounded-full"
                    style={{ backgroundColor: topEmotion.color }}
                  ></div>
                  <span className="text-sm font-medium text-blue-800 capitalize">
                    {topEmotion.emotion} ({(topEmotion.confidence * 100).toFixed(1)}%)
                  </span>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* App Features Explanation */}
        <div className="mb-8">
          <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl p-6 border border-blue-200">
            <h3 className="text-lg font-semibold text-blue-800 mb-3">
              🎉 Facial Emotion Recognition App Features
            </h3>
            <ul className="list-disc pl-5 space-y-1 text-blue-700">
              <li><b>Live Camera Analysis:</b> Capture your face in real time and get instant emotion, age, and gender predictions.</li>
              <li><b>Image Upload:</b> Upload any photo to analyze facial emotions, age, and gender.</li>
              <li><b>Emotion Visualization:</b> See a detailed breakdown of detected emotions with confidence bars.</li>
              <li><b>Session Statistics:</b> Track how many analyses you've performed and view top emotion confidence.</li>
              <li><b>Smart Device Integration:</b> Auto-adjust your environment (lights, music) based on detected emotion.</li>
              <li><b>Bias Reporting:</b> Report misclassifications and help improve model fairness.</li>
              <li><b>Avatar Mirror:</b> See a live animated avatar reflecting your detected emotion.</li>
              <li><b>Modern, Responsive UI:</b> Enjoy a clean, mobile-friendly interface built with React and Tailwind CSS.</li>
            </ul>
          </div>
        </div>

        {/* Main Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          {/* Camera Section */}
          <div className="space-y-6">
            <CameraCapture 
              onCapture={handleImageAnalysis}
              isAnalyzing={isAnalyzing}
            />
            
            <ImageUpload 
              onUpload={handleImageAnalysis}
              isAnalyzing={isAnalyzing}
            />
          </div>

          {/* Results Section */}
          <div className="space-y-6">
            {/* Age and Gender Display */}
            {(age !== null || gender) && !isAnalyzing && (
              <div className="bg-white rounded-xl shadow-lg p-4 flex flex-col items-center">
                {age !== null && (
                  <div className="text-lg font-semibold text-gray-700">Age: <span className="text-blue-600">{age}</span></div>
                )}
                {gender && (
                  <div className="text-lg font-semibold text-gray-700">Gender: <span className="text-blue-600 capitalize">{gender}</span></div>
                )}
              </div>
            )}
            <EmotionChart 
              emotions={emotions}
              isAnalyzing={isAnalyzing}
            />
            
            {/* Avatar Mirror */}
            <AvatarMirror currentEmotion={topEmotion} isAnalyzing={isAnalyzing} />
            {/* Smart Device Integration */}
            <SmartDeviceIntegration currentEmotion={topEmotion} />
            {/* Bias Reporting */}
            <BiasReporting currentEmotion={topEmotion} onReportSubmit={() => {}} />
            {/* Quick Stats */}
            <div className="bg-white rounded-xl shadow-lg p-6">
              <h3 className="text-xl font-semibold text-gray-800 mb-4">
                Session Statistics
              </h3>
              <div className="grid grid-cols-2 gap-4">
                <div className="text-center p-4 bg-blue-50 rounded-lg">
                  <div className="text-2xl font-bold text-blue-600">
                    {analysisCount}
                  </div>
                  <div className="text-sm text-blue-800">
                    Total Analyses
                  </div>
                </div>
                <div className="text-center p-4 bg-green-50 rounded-lg">
                  <div className="text-2xl font-bold text-green-600">
                    {topEmotion ? `${(topEmotion.confidence * 100).toFixed(0)}%` : '0%'}
                  </div>
                  <div className="text-sm text-green-800">
                    Top Confidence
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="text-center text-gray-500 text-sm">
          <p>
            Ready for production • Replace mock functions with your deep learning model
          </p>
        </div>
      </div>
    </div>
  );
}

export default App;