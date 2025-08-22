import React from 'react';

interface EmotionData {
  emotion: string;
  confidence: number;
  color: string;
}

interface EmotionChartProps {
  emotions: EmotionData[];
  isAnalyzing: boolean;
}

const EmotionChart: React.FC<EmotionChartProps> = ({ emotions, isAnalyzing }) => {
  return (
    <div className="bg-white rounded-xl shadow-lg p-6">
      <h3 className="text-xl font-semibold text-gray-800 mb-4">
        Emotion Analysis Results
      </h3>
      
      {isAnalyzing ? (
        <div className="space-y-4">
          <div className="flex items-center justify-center py-8">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
            <span className="ml-3 text-gray-600">Analyzing emotions...</span>
          </div>
        </div>
      ) : emotions.length > 0 ? (
        <div className="space-y-4">
          {emotions.map((emotion, index) => (
            <div key={emotion.emotion} className="space-y-2">
              <div className="flex justify-between items-center">
                <span className="text-sm font-medium text-gray-700 capitalize">
                  {emotion.emotion}
                </span>
                <span className="text-sm font-semibold text-gray-900">
                  {(emotion.confidence * 100).toFixed(1)}%
                </span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-3">
                <div
                  className="h-3 rounded-full transition-all duration-1000 ease-out"
                  style={{
                    width: `${emotion.confidence * 100}%`,
                    backgroundColor: emotion.color,
                    animationDelay: `${index * 100}ms`
                  }}
                ></div>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="text-center py-8 text-gray-500">
          <p>No analysis results yet</p>
          <p className="text-sm mt-2">Capture or upload an image to begin</p>
        </div>
      )}
    </div>
  );
};

export default EmotionChart;