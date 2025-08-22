import React, { useRef, useState, useCallback } from 'react';
import { Camera, CameraOff, Aperture as Capture } from 'lucide-react';

interface CameraCaptureProps {
  onCapture: (imageData: string) => void;
  isAnalyzing: boolean;
}

const CameraCapture: React.FC<CameraCaptureProps> = ({ onCapture, isAnalyzing }) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const startCamera = useCallback(async () => {
    try {
      setError(null);
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 }
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setIsStreaming(true);
      }
    } catch (err) {
      setError('Unable to access camera. Please ensure camera permissions are granted.');
      console.error('Camera access error:', err);
    }
  }, []);

  const stopCamera = useCallback(() => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
      setIsStreaming(false);
    }
  }, []);

  const captureImage = useCallback(() => {
    if (videoRef.current && canvasRef.current) {
      const canvas = canvasRef.current;
      const video = videoRef.current;
      const context = canvas.getContext('2d');
      
      if (context) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        context.drawImage(video, 0, 0);
        
        const imageData = canvas.toDataURL('image/jpeg');
        onCapture(imageData);
      }
    }
  }, [onCapture]);

  return (
    <div className="bg-white rounded-xl shadow-lg p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-xl font-semibold text-gray-800">Live Camera</h3>
        <div className="flex gap-2">
          {!isStreaming ? (
            <button
              onClick={startCamera}
              disabled={isAnalyzing}
              className="flex items-center gap-2 px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <Camera size={20} />
              Start Camera
            </button>
          ) : (
            <button
              onClick={stopCamera}
              disabled={isAnalyzing}
              className="flex items-center gap-2 px-4 py-2 bg-red-500 text-white rounded-lg hover:bg-red-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <CameraOff size={20} />
              Stop Camera
            </button>
          )}
        </div>
      </div>

      <div className="relative">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          className="w-full h-64 bg-gray-100 rounded-lg object-cover"
        />
        <canvas ref={canvasRef} className="hidden" />
        
        {error && (
          <div className="absolute inset-0 flex items-center justify-center bg-gray-100 rounded-lg">
            <div className="text-center">
              <CameraOff size={48} className="mx-auto text-gray-400 mb-2" />
              <p className="text-sm text-gray-600 max-w-xs">{error}</p>
            </div>
          </div>
        )}
      </div>

      {isStreaming && (
        <div className="mt-4 flex justify-center">
          <button
            onClick={captureImage}
            disabled={isAnalyzing}
            className="flex items-center gap-2 px-6 py-3 bg-emerald-500 text-white rounded-lg hover:bg-emerald-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            <Capture size={20} />
            {isAnalyzing ? 'Analyzing...' : 'Capture & Analyze'}
          </button>
        </div>
      )}
    </div>
  );
};

export default CameraCapture;