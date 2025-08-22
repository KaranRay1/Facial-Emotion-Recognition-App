import React, { useState, useEffect, useRef } from 'react';
import { Lightbulb, Volume2, Home, Wifi, WifiOff, Settings, Play, Pause, Bluetooth, BluetoothOff } from 'lucide-react';

interface EmotionData {
  emotion: string;
  confidence: number;
  color: string;
}

interface SmartDeviceIntegrationProps {
  currentEmotion: EmotionData | null;
}

interface DeviceState {
  lights: {
    enabled: boolean;
    brightness: number;
    color: string;
  };
  music: {
    enabled: boolean;
    volume: number;
    currentTrack: string;
    isPlaying: boolean;
  };
  connected: boolean;
  bluetoothDevice: BluetoothDevice | null;
}

const SmartDeviceIntegration: React.FC<SmartDeviceIntegrationProps> = ({ currentEmotion }) => {
  const [deviceState, setDeviceState] = useState<DeviceState>({
    lights: {
      enabled: false,
      brightness: 70,
      color: '#ffffff'
    },
    music: {
      enabled: false,
      volume: 50,
      currentTrack: 'None',
      isPlaying: false
    },
    connected: false,
    bluetoothDevice: null
  });

  const [showSettings, setShowSettings] = useState(false);
  const [scanning, setScanning] = useState(false);
  const [availableDevices, setAvailableDevices] = useState<BluetoothDevice[]>([]);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  // Initialize audio context
  useEffect(() => {
    audioRef.current = new Audio();
    audioRef.current.volume = deviceState.music.volume / 100;
    return () => {
      if (audioRef.current) {
        audioRef.current.pause();
      }
    };
  }, []);

  // Check if Web Bluetooth is supported
  const isBluetoothSupported = () => {
    return 'bluetooth' in navigator;
  };

  // Scan for Bluetooth devices
  const scanForDevices = async () => {
    if (!isBluetoothSupported()) {
      alert('Web Bluetooth is not supported in this browser. Try Chrome or Edge.');
      return;
    }

    setScanning(true);
    try {
      const device = await navigator.bluetooth.requestDevice({
        filters: [
          // Audio devices (speakers, headphones)
          { services: ['audio_service'] },
          { services: ['audio_sink'] },
          { services: ['audio_source'] },
          // Smart lights
          { services: ['light_service'] },
          { services: ['light_control'] },
          // Generic devices
          { services: ['generic_access'] }
        ],
        optionalServices: ['battery_service', 'device_information']
      });

      console.log('Bluetooth device selected:', device.name);
      setDeviceState(prev => ({
        ...prev,
        bluetoothDevice: device,
        connected: true
      }));

      // Listen for device disconnection
      device.addEventListener('gattserverdisconnected', () => {
        console.log('Bluetooth device disconnected');
        setDeviceState(prev => ({
          ...prev,
          connected: false,
          bluetoothDevice: null
        }));
      });

    } catch (error) {
      console.error('Bluetooth scan error:', error);
      if (error instanceof Error) {
        alert(`Bluetooth error: ${error.message}`);
      }
    } finally {
      setScanning(false);
    }
  };

  // Control Bluetooth audio device
  const controlBluetoothAudio = async (emotion: string, volume: number) => {
    if (!deviceState.bluetoothDevice) return;

    try {
      const server = await deviceState.bluetoothDevice.gatt?.connect();
      if (!server) return;

      // Try to find audio service
      const audioService = await server.getPrimaryService('audio_service') || 
                          await server.getPrimaryService('audio_sink') ||
                          await server.getPrimaryService('audio_source');

      if (audioService) {
        // Control volume
        const volumeCharacteristic = await audioService.getCharacteristic('volume');
        await volumeCharacteristic.writeValue(new Uint8Array([volume]));

        // Play emotion-based tone
        playEmotionTone(emotion, volume);
      }
    } catch (error) {
      console.error('Bluetooth audio control error:', error);
      // Fallback to browser audio
      playEmotionTone(emotion, volume);
    }
  };

  // Control Bluetooth lights
  const controlBluetoothLights = async (color: string, brightness: number) => {
    if (!deviceState.bluetoothDevice) return;

    try {
      const server = await deviceState.bluetoothDevice.gatt?.connect();
      if (!server) return;

      // Try to find light service
      const lightService = await server.getPrimaryService('light_service') ||
                          await server.getPrimaryService('light_control');

      if (lightService) {
        // Convert color to RGB values
        const rgb = hexToRgb(color);
        if (rgb) {
          const colorCharacteristic = await lightService.getCharacteristic('color');
          await colorCharacteristic.writeValue(new Uint8Array([rgb.r, rgb.g, rgb.b]));

          const brightnessCharacteristic = await lightService.getCharacteristic('brightness');
          await brightnessCharacteristic.writeValue(new Uint8Array([brightness]));
        }
      }
    } catch (error) {
      console.error('Bluetooth light control error:', error);
      // Fallback: just update UI
      setDeviceState(prev => ({
        ...prev,
        lights: { ...prev.lights, color, brightness }
      }));
    }
  };

  // Helper function to convert hex to RGB
  const hexToRgb = (hex: string) => {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ? {
      r: parseInt(result[1], 16),
      g: parseInt(result[2], 16),
      b: parseInt(result[3], 16)
    } : null;
  };

  // Play emotion-based tones using Web Audio API
  const playEmotionTone = (emotion: string, volume: number) => {
    const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
    const oscillator = audioContext.createOscillator();
    const gainNode = audioContext.createGain();

    oscillator.connect(gainNode);
    gainNode.connect(audioContext.destination);

    const frequencies = {
      happy: 440, sad: 220, angry: 330, fear: 110, surprised: 880, neutral: 330
    };

    oscillator.frequency.setValueAtTime(frequencies[emotion as keyof typeof frequencies] || 330, audioContext.currentTime);
    oscillator.type = 'sine';
    gainNode.gain.setValueAtTime(volume / 100, audioContext.currentTime);
    gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 3);

    oscillator.start(audioContext.currentTime);
    oscillator.stop(audioContext.currentTime + 3);

    setDeviceState(prev => ({
      ...prev,
      music: {
        ...prev.music,
        isPlaying: true,
        currentTrack: `${emotion.charAt(0).toUpperCase() + emotion.slice(1)} Tone`
      }
    }));

    // Stop playing after 3 seconds
    setTimeout(() => {
      setDeviceState(prev => ({
        ...prev,
        music: { ...prev.music, isPlaying: false, currentTrack: 'None' }
      }));
    }, 3000);
  };

  // Auto-adjust devices based on emotion
  useEffect(() => {
    if (currentEmotion && deviceState.connected) {
      const emotion = currentEmotion.emotion.toLowerCase();
      
      // Control Bluetooth lights
      if (deviceState.lights.enabled) {
        const colors = {
          happy: '#FFD700', sad: '#4169E1', angry: '#FF6B6B', 
          fear: '#DDA0DD', surprised: '#FFA500', neutral: '#F0F8FF'
        };
        const brightness = {
          happy: 90, sad: 40, angry: 60, 
          fear: 30, surprised: 80, neutral: 60
        };
        
        controlBluetoothLights(colors[emotion as keyof typeof colors], brightness[emotion as keyof typeof brightness]);
      }

      // Control Bluetooth audio
      if (deviceState.music.enabled) {
        controlBluetoothAudio(emotion, deviceState.music.volume);
      }
    }
  }, [currentEmotion, deviceState.connected, deviceState.lights.enabled, deviceState.music.enabled]);

  const getEmotionRecommendation = (emotion: string) => {
    const recommendations = {
      happy: "Bright, warm lighting and upbeat music to enhance your joy!",
      sad: "Soft blue lighting and calming music to provide comfort.",
      angry: "Gentle lighting and meditation sounds to help you relax.",
      fear: "Warm, dim lighting and peaceful sounds for reassurance.",
      surprised: "Bright, energetic lighting to match your excitement!",
      neutral: "Balanced lighting and ambient sounds for focus."
    };
    return recommendations[emotion?.toLowerCase() as keyof typeof recommendations] || "Adjusting environment to match your mood...";
  };

  return (
    <div className="bg-white rounded-xl shadow-lg p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-xl font-semibold text-gray-800">Bluetooth Device Integration</h3>
        <div className="flex items-center space-x-2">
          <button onClick={() => setShowSettings(!showSettings)} className="p-2 text-gray-500 hover:text-gray-700">
            <Settings size={20} />
          </button>
          <button onClick={scanForDevices} disabled={scanning} className={`flex items-center space-x-2 px-3 py-1 rounded-full text-sm font-medium ${
            deviceState.connected 
              ? 'bg-green-100 text-green-800' 
              : scanning
              ? 'bg-yellow-100 text-yellow-800'
              : 'bg-blue-100 text-blue-800'
          }`}>
            {deviceState.connected ? <Bluetooth size={16} /> : scanning ? <Bluetooth size={16} /> : <BluetoothOff size={16} />}
            <span>{deviceState.connected ? deviceState.bluetoothDevice?.name || 'Connected' : scanning ? 'Scanning...' : 'Connect Device'}</span>
          </button>
        </div>
      </div>

      {showSettings && (
        <div className="mb-4 p-4 bg-gray-50 rounded-lg">
          <h4 className="font-medium text-gray-800 mb-2">Bluetooth Setup</h4>
          <p className="text-sm text-gray-600 mb-2">Supported devices:</p>
          <ul className="text-xs text-gray-600 space-y-1">
            <li>• Bluetooth speakers and headphones</li>
            <li>• Smart LED lights with Bluetooth</li>
            <li>• Smart plugs with Bluetooth</li>
            <li>• Any device with audio_service or light_service</li>
          </ul>
          <p className="text-xs text-gray-500 mt-2">Note: Requires HTTPS or localhost for Web Bluetooth API</p>
        </div>
      )}

      {!deviceState.connected ? (
        <div className="text-center py-8 text-gray-500">
          <Bluetooth size={48} className="mx-auto mb-4 text-gray-300" />
          <p className="text-lg font-medium">Connect Bluetooth Device</p>
          <p className="text-sm mt-2">Click "Connect Device" to scan for nearby Bluetooth speakers, lights, or smart devices.</p>
          {!isBluetoothSupported() && (
            <p className="text-xs text-red-500 mt-2">Web Bluetooth not supported. Use Chrome or Edge browser.</p>
          )}
        </div>
      ) : (
        <div className="space-y-6">
          {currentEmotion && (
            <div className="bg-blue-50 rounded-lg p-4">
              <h4 className="font-medium text-blue-800 mb-2">Environment Recommendation</h4>
              <p className="text-sm text-blue-700">{getEmotionRecommendation(currentEmotion.emotion)}</p>
            </div>
          )}

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="border rounded-lg p-4">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center space-x-2">
                  <Lightbulb size={20} className="text-yellow-500" />
                  <span className="font-medium">Smart Lights</span>
                </div>
                <label className="relative inline-flex items-center cursor-pointer">
                  <input type="checkbox" checked={deviceState.lights.enabled} onChange={(e) => setDeviceState(prev => ({
                    ...prev, lights: { ...prev.lights, enabled: e.target.checked }
                  }))} className="sr-only peer" />
                  <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                </label>
              </div>
              
              {deviceState.lights.enabled && (
                <div className="space-y-3">
                  <div>
                    <label className="text-xs text-gray-600">Brightness: {deviceState.lights.brightness}%</label>
                    <input type="range" min="10" max="100" value={deviceState.lights.brightness} onChange={(e) => setDeviceState(prev => ({
                      ...prev, lights: { ...prev.lights, brightness: parseInt(e.target.value) }
                    }))} className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer" />
                  </div>
                  <div className="flex items-center space-x-2">
                    <div className="w-6 h-6 rounded border-2 border-gray-300" style={{ backgroundColor: deviceState.lights.color }}></div>
                    <span className="text-sm text-gray-600">Auto-adjusted for {currentEmotion?.emotion || 'neutral'}</span>
                  </div>
                </div>
              )}
            </div>

            <div className="border rounded-lg p-4">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center space-x-2">
                  <Volume2 size={20} className="text-blue-500" />
                  <span className="font-medium">Smart Audio</span>
                </div>
                <label className="relative inline-flex items-center cursor-pointer">
                  <input type="checkbox" checked={deviceState.music.enabled} onChange={(e) => setDeviceState(prev => ({
                    ...prev, music: { ...prev.music, enabled: e.target.checked }
                  }))} className="sr-only peer" />
                  <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                </label>
              </div>
              
              {deviceState.music.enabled && (
                <div className="space-y-3">
                  <div>
                    <label className="text-xs text-gray-600">Volume: {deviceState.music.volume}%</label>
                    <input type="range" min="0" max="100" value={deviceState.music.volume} onChange={(e) => setDeviceState(prev => ({
                      ...prev, music: { ...prev.music, volume: parseInt(e.target.value) }
                    }))} className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer" />
                  </div>
                  <div className="flex items-center justify-between">
                    <div className="text-sm text-gray-600">
                      <span className="font-medium">Now Playing:</span> {deviceState.music.currentTrack}
                    </div>
                    {deviceState.music.isPlaying && (
                      <div className="p-1 text-green-500">
                        <Play size={16} />
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>

          <div className="bg-gray-50 rounded-lg p-4">
            <h4 className="font-medium text-gray-800 mb-2">Device Status</h4>
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div className="flex justify-between">
                <span>Connected Device:</span>
                <span className="text-green-600">{deviceState.bluetoothDevice?.name || 'Unknown'}</span>
              </div>
              <div className="flex justify-between">
                <span>Smart Lights:</span>
                <span className={deviceState.lights.enabled ? 'text-green-600' : 'text-gray-500'}>
                  {deviceState.lights.enabled ? 'Active' : 'Inactive'}
                </span>
              </div>
              <div className="flex justify-between">
                <span>Smart Audio:</span>
                <span className={deviceState.music.enabled ? 'text-green-600' : 'text-gray-500'}>
                  {deviceState.music.enabled ? 'Active' : 'Inactive'}
                </span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default SmartDeviceIntegration;