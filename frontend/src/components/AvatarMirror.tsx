import React, { useEffect, useState } from 'react';
import { Smile, Frown, Angry, Sunrise as Surprise, Meh, Heart as Fearful, Asterisk as Disgusted } from 'lucide-react';

interface EmotionData {
  emotion: string;
  confidence: number;
  color: string;
}

interface AvatarMirrorProps {
  currentEmotion: EmotionData | null;
  isAnalyzing: boolean;
}

const AvatarMirror: React.FC<AvatarMirrorProps> = ({ currentEmotion, isAnalyzing }) => {
  const [animationClass, setAnimationClass] = useState('');

  useEffect(() => {
    if (currentEmotion) {
      setAnimationClass('animate-pulse');
      const timer = setTimeout(() => setAnimationClass(''), 1000);
      return () => clearTimeout(timer);
    }
  }, [currentEmotion]);

  const getEmotionIcon = (emotion: string) => {
    const iconProps = { size: 120, className: `transition-all duration-500 ${animationClass}` };
    
    switch (emotion?.toLowerCase()) {
      case 'happy':
        return <Smile {...iconProps} className={`${iconProps.className} text-green-500`} />;
      case 'sad':
        return <Frown {...iconProps} className={`${iconProps.className} text-blue-500`} />;
      case 'angry':
        return <Angry {...iconProps} className={`${iconProps.className} text-red-500`} />;
      case 'surprised':
        return <Surprise {...iconProps} className={`${iconProps.className} text-orange-500`} />;
      case 'fear':
        return <Fearful {...iconProps} className={`${iconProps.className} text-purple-500`} />;
      case 'disgust':
        return <Disgusted {...iconProps} className={`${iconProps.className} text-pink-500`} />;
      default:
        return <Meh {...iconProps} className={`${iconProps.className} text-gray-500`} />;
    }
  };

  const getEmotionMessage = (emotion: string) => {
    const messages = {
      happy: "I'm feeling joyful! 😊",
      sad: "I sense some sadness... 😢",
      angry: "There's some anger here! 😠",
      surprised: "Wow, that's surprising! 😲",
      fear: "I detect some fear... 😨",
      disgust: "Something seems off... 🤢",
      neutral: "I'm feeling neutral 😐"
    };
    return messages[emotion?.toLowerCase() as keyof typeof messages] || "Analyzing emotions...";
  };

  return (
    <div className="bg-white rounded-xl shadow-lg p-6">
      <h3 className="text-xl font-semibold text-gray-800 mb-4 text-center">
        Emotion Avatar Mirror
      </h3>
      
      <div className="flex flex-col items-center space-y-4">
        {/* Avatar Face */}
        <div className="relative">
          <div className="w-40 h-40 rounded-full bg-gradient-to-br from-blue-100 to-indigo-200 flex items-center justify-center border-4 border-white shadow-lg">
            {isAnalyzing ? (
              <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-500"></div>
            ) : (
              getEmotionIcon(currentEmotion?.emotion || 'neutral')
            )}
          </div>
          
          {/* Emotion Glow Effect */}
          {currentEmotion && (
            <div 
              className="absolute inset-0 rounded-full opacity-30 animate-ping"
              style={{ backgroundColor: currentEmotion.color }}
            ></div>
          )}
        </div>

        {/* Emotion Message */}
        <div className="text-center">
          <p className="text-lg font-medium text-gray-700 mb-2">
            {isAnalyzing ? "Reading your emotions..." : getEmotionMessage(currentEmotion?.emotion || 'neutral')}
          </p>
          
          {currentEmotion && (
            <div className="flex items-center justify-center space-x-2">
              <div 
                className="w-3 h-3 rounded-full"
                style={{ backgroundColor: currentEmotion.color }}
              ></div>
              <span className="text-sm text-gray-600 capitalize">
                {currentEmotion.emotion} ({(currentEmotion.confidence * 100).toFixed(1)}% confident)
              </span>
            </div>
          )}
        </div>

        {/* Emotion Intensity Bar */}
        {currentEmotion && (
          <div className="w-full max-w-xs">
            <div className="flex justify-between text-xs text-gray-500 mb-1">
              <span>Emotion Intensity</span>
              <span>{(currentEmotion.confidence * 100).toFixed(0)}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className="h-2 rounded-full transition-all duration-1000 ease-out"
                style={{
                  width: `${currentEmotion.confidence * 100}%`,
                  backgroundColor: currentEmotion.color
                }}
              ></div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default AvatarMirror;