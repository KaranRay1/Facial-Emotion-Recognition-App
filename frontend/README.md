# Facial Emotion Recognition System UI

A modern, production-ready web interface for facial emotion recognition using deep learning models. Built with React, TypeScript, and Tailwind CSS.

## 🌟 Features

- **Real-time Camera Capture**: Live webcam integration with emotion detection
- **Image Upload**: Drag-and-drop or click-to-upload image analysis
- **Emotion Visualization**: Animated confidence bars for emotion probabilities
- **Session Statistics**: Track analysis count and confidence scores
- **Responsive Design**: Optimized for desktop, tablet, and mobile devices
- **Modern UI**: Clean, professional interface with smooth animations
- **Model Integration Ready**: Easy integration points for your deep learning models

## 🚀 Live Demo

Visit the live application: [https://shiny-crepe-effe42.netlify.app](https://shiny-crepe-effe42.netlify.app)

## 🛠️ Technology Stack

- **Frontend**: React 18 with TypeScript
- **Styling**: Tailwind CSS
- **Icons**: Lucide React
- **Build Tool**: Vite
- **Deployment**: Netlify

## 📦 Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd facial-emotion-recognition-ui
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

4. Open your browser and navigate to `http://localhost:5173`

## 🔧 Model Integration

The application includes a mock emotion analysis function that you can replace with your actual deep learning model. Here are the integration options:

### Option 1: TensorFlow.js (Client-side)
```typescript
import * as tf from '@tensorflow/tfjs';

export const analyzeEmotion = async (imageData: string): Promise<EmotionData[]> => {
  // Load your pre-trained model
  const model = await tf.loadLayersModel('/path/to/your/model.json');
  
  // Preprocess image
  const tensor = tf.browser.fromPixels(imageElement)
    .resizeNearestNeighbor([224, 224])
    .expandDims(0)
    .div(255.0);
  
  // Run inference
  const predictions = await model.predict(tensor) as tf.Tensor;
  const results = await predictions.data();
  
  // Format results
  return formatEmotionResults(results);
};
```

### Option 2: API Integration (Server-side)
```typescript
export const analyzeEmotion = async (imageData: string): Promise<EmotionData[]> => {
  const response = await fetch('/api/analyze-emotion', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ image: imageData }),
  });
  
  const results = await response.json();
  return results.emotions;
};
```

### Option 3: Cloud Services
```typescript
// Example with AWS Rekognition
import AWS from 'aws-sdk';

const rekognition = new AWS.Rekognition();

export const analyzeEmotion = async (imageData: string): Promise<EmotionData[]> => {
  const params = {
    Image: {
      Bytes: Buffer.from(imageData.split(',')[1], 'base64')
    },
    Attributes: ['ALL']
  };
  
  const result = await rekognition.detectFaces(params).promise();
  return formatRekognitionResults(result);
};
```

## 📁 Project Structure

```
src/
├── components/
│   ├── CameraCapture.tsx      # Webcam integration component
│   ├── ImageUpload.tsx        # File upload component
│   ├── EmotionChart.tsx       # Results visualization
│   └── ModelIntegration.tsx   # Model integration utilities
├── App.tsx                    # Main application component
├── main.tsx                   # Application entry point
└── index.css                  # Global styles
```

## 🎨 Emotion Categories

The system recognizes the following emotions:

- **Happy** 😊 - Green (#10B981)
- **Sad** 😢 - Blue (#3B82F6)
- **Angry** 😠 - Red (#EF4444)
- **Surprised** 😲 - Orange (#F59E0B)
- **Neutral** 😐 - Gray (#6B7280)
- **Fear** 😨 - Purple (#8B5CF6)
- **Disgust** 🤢 - Pink (#EC4899)

## 🔒 Privacy & Security

- **Local Processing**: Camera data is processed locally in the browser
- **No Data Storage**: Images are not stored or transmitted unless explicitly configured
- **Secure Connections**: All API calls use HTTPS
- **Permission-based**: Camera access requires explicit user permission

## 📱 Browser Compatibility

- Chrome 60+
- Firefox 55+
- Safari 11+
- Edge 79+

**Note**: Camera functionality requires HTTPS in production environments.

## 🚀 Deployment

### Netlify (Recommended)
```bash
npm run build
# Deploy the dist/ folder to Netlify
```

### Vercel
```bash
npm install -g vercel
vercel --prod
```

### Manual Deployment
```bash
npm run build
# Upload the dist/ folder to your hosting provider
```

## 🧪 Development

### Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint

### Environment Variables

Create a `.env` file for environment-specific configuration:

```env
VITE_API_URL=https://your-api-endpoint.com
VITE_MODEL_PATH=/path/to/your/model
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [React](https://reactjs.org/) - UI framework
- [Tailwind CSS](https://tailwindcss.com/) - Styling framework
- [Lucide React](https://lucide.dev/) - Icon library
- [Vite](https://vitejs.dev/) - Build tool

## 📞 Support

For support and questions:

- Create an issue in the repository
- Contact the development team
- Check the documentation

---

**Ready for production** • Replace mock functions with your deep learning model integration