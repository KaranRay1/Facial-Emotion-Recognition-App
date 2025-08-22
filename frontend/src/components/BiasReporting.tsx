import React, { useState } from 'react';
import { AlertTriangle, CheckCircle, X, Send } from 'lucide-react';

interface EmotionData {
  emotion: string;
  confidence: number;
  color: string;
}

interface BiasReportingProps {
  currentEmotion: EmotionData | null;
  onReportSubmit: (report: BiasReport) => void;
}

interface BiasReport {
  predictedEmotion: string;
  actualEmotion: string;
  confidence: number;
  userDemographics: {
    ageGroup: string;
    skinTone: string;
    lighting: string;
    gender: string;
  };
  feedback: string;
  timestamp: Date;
}

const BiasReporting: React.FC<BiasReportingProps> = ({ currentEmotion, onReportSubmit }) => {
  const [showReportForm, setShowReportForm] = useState(false);
  const [reportData, setReportData] = useState<Partial<BiasReport>>({
    actualEmotion: '',
    userDemographics: {
      ageGroup: '',
      skinTone: '',
      lighting: '',
      gender: ''
    },
    feedback: ''
  });

  const emotions = ['happy', 'sad', 'angry', 'surprised', 'fear', 'disgust', 'neutral'];
  const ageGroups = ['18-25', '26-35', '36-45', '46-55', '56-65', '65+'];
  const skinTones = ['Very Light', 'Light', 'Medium Light', 'Medium', 'Medium Dark', 'Dark', 'Very Dark'];
  const lightingConditions = ['Bright Natural', 'Dim Natural', 'Bright Artificial', 'Dim Artificial', 'Mixed Lighting'];
  const genderOptions = ['Male', 'Female', 'Non-binary', 'Prefer not to say'];

  const handleSubmitReport = () => {
    if (currentEmotion && reportData.actualEmotion) {
      const report: BiasReport = {
        predictedEmotion: currentEmotion.emotion,
        actualEmotion: reportData.actualEmotion,
        confidence: currentEmotion.confidence,
        userDemographics: reportData.userDemographics as BiasReport['userDemographics'],
        feedback: reportData.feedback || '',
        timestamp: new Date()
      };
      
      onReportSubmit(report);
      setShowReportForm(false);
      setReportData({
        actualEmotion: '',
        userDemographics: {
          ageGroup: '',
          skinTone: '',
          lighting: '',
          gender: ''
        },
        feedback: ''
      });
    }
  };

  const updateDemographics = (field: string, value: string) => {
    setReportData(prev => ({
      ...prev,
      userDemographics: {
        ...prev.userDemographics,
        [field]: value
      }
    }));
  };

  if (!currentEmotion) {
    return (
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h3 className="text-xl font-semibold text-gray-800 mb-4">Bias Reporting</h3>
        <div className="text-center py-8 text-gray-500">
          <AlertTriangle size={48} className="mx-auto mb-4 text-gray-300" />
          <p>No emotion prediction to report</p>
          <p className="text-sm mt-2">Analyze an image first to provide feedback</p>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-xl shadow-lg p-6">
      <h3 className="text-xl font-semibold text-gray-800 mb-4">Bias Reporting & Feedback</h3>
      
      {!showReportForm ? (
        <div className="space-y-4">
          <div className="bg-blue-50 rounded-lg p-4">
            <h4 className="font-medium text-blue-800 mb-2">Current Prediction</h4>
            <div className="flex items-center justify-between">
              <span className="capitalize text-blue-700">
                {currentEmotion.emotion} ({(currentEmotion.confidence * 100).toFixed(1)}% confident)
              </span>
              <div 
                className="w-4 h-4 rounded-full"
                style={{ backgroundColor: currentEmotion.color }}
              ></div>
            </div>
          </div>

          <div className="text-center space-y-4">
            <p className="text-gray-600">
              Was this prediction accurate? Your feedback helps improve the model for everyone.
            </p>
            
            <div className="flex justify-center space-x-4">
              <button
                onClick={() => {/* Handle correct prediction */}}
                className="flex items-center space-x-2 px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 transition-colors"
              >
                <CheckCircle size={20} />
                <span>Correct</span>
              </button>
              
              <button
                onClick={() => setShowReportForm(true)}
                className="flex items-center space-x-2 px-4 py-2 bg-orange-500 text-white rounded-lg hover:bg-orange-600 transition-colors"
              >
                <AlertTriangle size={20} />
                <span>Report Issue</span>
              </button>
            </div>
          </div>

          <div className="bg-yellow-50 rounded-lg p-4 text-sm">
            <h5 className="font-medium text-yellow-800 mb-2">Why Report Bias?</h5>
            <ul className="text-yellow-700 space-y-1">
              <li>• Helps identify model weaknesses across different demographics</li>
              <li>• Improves accuracy for underrepresented groups</li>
              <li>• Ensures fair and inclusive emotion recognition</li>
              <li>• Your data is anonymized and used only for model improvement</li>
            </ul>
          </div>
        </div>
      ) : (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h4 className="font-medium text-gray-800">Report Prediction Issue</h4>
            <button
              onClick={() => setShowReportForm(false)}
              className="text-gray-500 hover:text-gray-700"
            >
              <X size={20} />
            </button>
          </div>

          <div className="space-y-4">
            {/* Actual Emotion */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                What emotion were you actually expressing?
              </label>
              <select
                value={reportData.actualEmotion}
                onChange={(e) => setReportData(prev => ({ ...prev, actualEmotion: e.target.value }))}
                className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value="">Select actual emotion</option>
                {emotions.map(emotion => (
                  <option key={emotion} value={emotion} className="capitalize">
                    {emotion}
                  </option>
                ))}
              </select>
            </div>

            {/* Demographics */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Age Group</label>
                <select
                  value={reportData.userDemographics?.ageGroup || ''}
                  onChange={(e) => updateDemographics('ageGroup', e.target.value)}
                  className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  <option value="">Select age group</option>
                  {ageGroups.map(age => (
                    <option key={age} value={age}>{age}</option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Skin Tone</label>
                <select
                  value={reportData.userDemographics?.skinTone || ''}
                  onChange={(e) => updateDemographics('skinTone', e.target.value)}
                  className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  <option value="">Select skin tone</option>
                  {skinTones.map(tone => (
                    <option key={tone} value={tone}>{tone}</option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Lighting Condition</label>
                <select
                  value={reportData.userDemographics?.lighting || ''}
                  onChange={(e) => updateDemographics('lighting', e.target.value)}
                  className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  <option value="">Select lighting</option>
                  {lightingConditions.map(lighting => (
                    <option key={lighting} value={lighting}>{lighting}</option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Gender (Optional)</label>
                <select
                  value={reportData.userDemographics?.gender || ''}
                  onChange={(e) => updateDemographics('gender', e.target.value)}
                  className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  <option value="">Select gender</option>
                  {genderOptions.map(gender => (
                    <option key={gender} value={gender}>{gender}</option>
                  ))}
                </select>
              </div>
            </div>

            {/* Additional Feedback */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Additional Feedback (Optional)
              </label>
              <textarea
                value={reportData.feedback}
                onChange={(e) => setReportData(prev => ({ ...prev, feedback: e.target.value }))}
                placeholder="Any additional context about the misclassification..."
                rows={3}
                className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>

            {/* Submit Button */}
            <button
              onClick={handleSubmitReport}
              disabled={!reportData.actualEmotion}
              className="w-full flex items-center justify-center space-x-2 px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <Send size={20} />
              <span>Submit Report</span>
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default BiasReporting;