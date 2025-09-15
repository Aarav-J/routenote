"use client";

import { useState, useRef, ChangeEvent } from "react";
import Image from "next/image";
import ClimbingImageAnalyzer from "./components/ClimbingImageAnalyzer";
import { AnalysisResult } from "./utils/imageProcessing";

export default function Home() {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [imageObjectUrl, setImageObjectUrl] = useState<string | null>(null);
  const [analysisData, setAnalysisData] = useState<AnalysisResult | null>(null);
  const [legendImage, setLegendImage] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    setError(null);
    const file = e.target.files?.[0];
    if (!file) return;

    // Check if file is an image
    if (!file.type.startsWith('image/')) {
      setError('Please select an image file');
      return;
    }

    // Create a preview of the selected image
    const reader = new FileReader();
    reader.onload = (e) => {
      setSelectedImage(e.target?.result as string);
      
      // Create an object URL for the file (for the analyzer component)
      if (imageObjectUrl) {
        URL.revokeObjectURL(imageObjectUrl); // Clean up previous object URL
      }
      setImageObjectUrl(URL.createObjectURL(file));
      
      // Reset previous analysis
      setAnalysisData(null);
      setLegendImage(null);
    };
    reader.readAsDataURL(file);
  };

  // const handleUpload = async () => {
  //   if (!selectedImage) {
  //     setError('Please select an image first');
  //     return;
  //   }

  //   setIsLoading(true);
  //   setError(null);

  //   try {
  //     // Create a form to send the image to the backend
  //     const formData = new FormData();
  //     const file = fileInputRef.current?.files?.[0];
      
  //     if (!file) {
  //       setError('No file selected');
  //       setIsLoading(false);
  //       return;
  //     }
      
  //     formData.append('image', file);
      
  //     const response = await fetch('/api/process-image', {
  //       method: 'POST',
  //       body: formData,
  //     });
      
  //     if (!response.ok) {
  //       const errorData = await response.json();
  //       throw new Error(errorData.error || 'Failed to process image');
  //     }
      
  //     const data = await response.json();
  //     setProcessedImage(data.processedImageUrl);
  //     setLegendImage(data.legendImageUrl);
  //     setIsLoading(false);
  //   } catch (err) {
  //     setError('Failed to process the image. Please try again.');
  //     setIsLoading(false);
  //   }
  // };
  const handleUpload = async () => {
  if (!selectedImage) {
    setError('Please select an image first');
    return;
  }

  setIsLoading(true);
  setError(null);

  try {
    // Create a form to send the image to the backend
    const formData = new FormData();
    const file = fileInputRef.current?.files?.[0];
    
    if (!file) {
      setError('No file selected');
      setIsLoading(false);
      return;
    }
    
    formData.append('file', file); // FastAPI typically uses 'file' as the form field name
    
    const response = await fetch('http://localhost:8000/api/analyze', {
      method: 'POST',
      body: formData,
      // No need to set Content-Type header as the browser will set it correctly with boundary for FormData
    });
    
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Failed to process image');
    }
    
    // Parse the analysis data
    const data = await response.json();
    setAnalysisData(data);
    
    // If the API returns a legendImage URL
    if (data.legendImageUrl) {
      setLegendImage(data.legendImageUrl);
    }
    
    setIsLoading(false);
  } catch (err) {
    console.error('Error processing image:', err);
    setError('Failed to process the image. Please try again.');
    setIsLoading(false);
  }
};
  // Save notes to localStorage
  const handleSaveNotes = (notes: Record<string, string>) => {
    console.log('Saving notes:', notes);
    localStorage.setItem('climbingWallNotes', JSON.stringify(notes));
  };

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-gray-100 transition-colors duration-300">
      <div className="container mx-auto px-4 py-8">
        <header className="text-center mb-12">
          <h1 className="text-4xl font-bold mb-2">ClimbColor Analyzer</h1>
          <p className="text-lg text-gray-600 dark:text-gray-400">
            Upload climbing wall images to detect and analyze hold colors
          </p>
        </header>

        <div className="max-w-4xl mx-auto">
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 mb-8">
            <div className="mb-6">
              <h2 className="text-xl font-semibold mb-4">Upload Image</h2>
              
              <div className="flex flex-col items-center">
                <div 
                  className="w-full border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-lg p-6 text-center cursor-pointer hover:border-gray-400 dark:hover:border-gray-500 transition-colors"
                  onClick={() => fileInputRef.current?.click()}
                >
                  {selectedImage ? (
                    <div className="relative w-full max-w-md mx-auto aspect-video">
                      <img 
                        src={selectedImage} 
                        alt="Selected climbing wall" 
                        className="rounded-lg object-cover w-full h-full"
                      />
                    </div>
                  ) : (
                    <div className="py-8">
                      <svg 
                        className="mx-auto h-12 w-12 text-gray-400 dark:text-gray-500" 
                        xmlns="http://www.w3.org/2000/svg" 
                        fill="none" 
                        viewBox="0 0 24 24" 
                        stroke="currentColor"
                      >
                        <path 
                          strokeLinecap="round" 
                          strokeLinejoin="round" 
                          strokeWidth={2} 
                          d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" 
                        />
                      </svg>
                      <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
                        Click to select or drag and drop a climbing wall image
                      </p>
                      <p className="mt-1 text-xs text-gray-400 dark:text-gray-500">
                        PNG, JPG, WEBP up to 10MB
                      </p>
                    </div>
                  )}
                  <input
                    ref={fileInputRef}
                    type="file"
                    className="hidden"
                    onChange={handleFileChange}
                    accept="image/*"
                  />
                </div>
                
                {error && (
                  <p className="text-red-500 mt-2 text-sm">{error}</p>
                )}
                
                <button
                  onClick={handleUpload}
                  disabled={!selectedImage || isLoading}
                  className={`mt-4 px-6 py-2 rounded-full font-medium text-white 
                    ${!selectedImage || isLoading 
                      ? 'bg-gray-400 dark:bg-gray-700 cursor-not-allowed' 
                      : 'bg-blue-600 hover:bg-blue-700 dark:bg-blue-500 dark:hover:bg-blue-600'}
                    transition-colors duration-300`}
                >
                  {isLoading ? (
                    <span className="flex items-center">
                      <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                      Processing...
                    </span>
                  ) : 'Analyze Image'}
                </button>
              </div>
            </div>
          </div>
          
          {analysisData && imageObjectUrl && (
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
              <h2 className="text-xl font-semibold mb-4">Analysis Results</h2>
              
              <div className="space-y-8">
                {/* Use our new component for image analysis */}
                <ClimbingImageAnalyzer 
                  originalImage={imageObjectUrl} 
                  analysisData={analysisData} 
                  isLoading={isLoading}
                  onSaveNotes={handleSaveNotes}
                />
                
                {legendImage && (
                  <div>
                    <h3 className="text-lg font-medium text-gray-700 dark:text-gray-300 mb-2">Color Clusters</h3>
                    <div className="relative w-full">
                      <img 
                        src={legendImage}
                        alt="Color cluster legend" 
                        className="rounded-lg object-contain w-full max-h-[400px] mx-auto bg-white dark:bg-gray-700 p-2"
                      />
                    </div>
                    <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">
                      Legend showing the different color clusters and the number of holds in each
                    </p>
                  </div>
                )}
                
                
                
                <div className="flex justify-center">
                  <button
                    onClick={() => {
                      setSelectedImage(null);
                      setImageObjectUrl(prev => {
                        if (prev) URL.revokeObjectURL(prev);
                        return null;
                      });
                      setAnalysisData(null);
                      setLegendImage(null);
                      if (fileInputRef.current) {
                        fileInputRef.current.value = '';
                      }
                    }}
                    className="px-6 py-2 rounded-full font-medium text-white bg-blue-600 hover:bg-blue-700 dark:bg-blue-500 dark:hover:bg-blue-600 transition-colors duration-300"
                  >
                    Analyze Another Image
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
      
      <footer className="mt-12 py-6 border-t border-gray-200 dark:border-gray-800">
        <div className="container mx-auto px-4 text-center text-gray-500 dark:text-gray-400 text-sm">
          ClimbColor Analyzer &copy; {new Date().getFullYear()}
        </div>
      </footer>
    </div>
  );
}
