"use client";

import { useEffect, useRef, useState } from "react";
import { AnalysisResult, Cluster, drawBoundingBoxes, isDarkColor } from "../utils/imageProcessing";

interface ClimbingImageProps {
  originalImage: string;
  analysisData: AnalysisResult | null;
  isLoading?: boolean;
}

export default function ClimbingImageAnalyzer({ originalImage, analysisData, isLoading = false }: ClimbingImageProps) {
  const [selectedCluster, setSelectedCluster] = useState<number | null>(null);
  const [imageLoaded, setImageLoaded] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  
  // Effect to draw bounding boxes when data or selected cluster changes
  useEffect(() => {
    if (imageLoaded && analysisData) {
      drawBoundingBoxes(
        canvasRef.current, 
        imageRef.current, 
        analysisData, 
        selectedCluster
      );
    }
  }, [analysisData, selectedCluster, imageLoaded]);
  
  // Handle cluster selection toggle
  const handleClusterClick = (clusterId: number) => {
    setSelectedCluster(selectedCluster === clusterId ? null : clusterId);
  };

  return (
    <div className="space-y-8">
      <div>
        <h3 className="text-lg font-medium text-gray-700 dark:text-gray-300 mb-2">Analyzed Image</h3>
        <div className="relative w-full mx-auto" style={{ maxWidth: '100%', textAlign: 'center' }}>
          {/* The main image */}
          <div style={{ position: 'relative', display: 'inline-block', maxWidth: '100%' }}>
            <img 
              ref={imageRef}
              src={originalImage}
              alt="Climbing wall with hold detection"
              className="rounded-lg max-h-[600px] max-w-full"
              style={{ display: 'block' }}
              onLoad={() => setImageLoaded(true)}
            />
            
            {/* Canvas overlay for bounding boxes */}
            <canvas
              ref={canvasRef}
              className="absolute top-0 left-0 pointer-events-none rounded-lg"
              style={{ position: 'absolute' }}
            />
          </div>
          
          {/* Loading overlay */}
          {isLoading && (
            <div className="absolute inset-0 bg-gray-900 bg-opacity-50 flex items-center justify-center rounded-lg">
              <div className="text-white flex flex-col items-center">
                <svg className="animate-spin h-10 w-10 mb-2" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                <span>Analyzing image...</span>
              </div>
            </div>
          )}
        </div>
        <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">
          Climbing holds are grouped by color similarity and marked with their cluster number
        </p>
      </div>
      
      {/* Cluster selection buttons */}
      {analysisData && analysisData.clusters && analysisData.clusters.length > 0 && (
        <div>
          <h3 className="text-lg font-medium text-gray-700 dark:text-gray-300 mb-2">Color Clusters</h3>
          <div className="flex flex-wrap gap-2 mt-3">
            {analysisData.clusters.map((cluster: Cluster) => (
              <button
                key={cluster.cluster_id}
                onClick={() => handleClusterClick(cluster.cluster_id)}
                className={`px-3 py-1 rounded-full flex items-center ${
                  selectedCluster === cluster.cluster_id 
                    ? 'ring-2 ring-offset-2 ring-blue-500 dark:ring-offset-gray-800' 
                    : ''
                }`}
                style={{
                  backgroundColor: `rgb(${cluster.rgb_color[0]}, ${cluster.rgb_color[1]}, ${cluster.rgb_color[2]})`,
                  color: isDarkColor(cluster.rgb_color) ? 'white' : 'black'
                }}
              >
                <span className="font-medium">Cluster {cluster.cluster_id}</span>
                <span className="ml-2 bg-white bg-opacity-30 dark:bg-black dark:bg-opacity-30 text-xs rounded-full px-2">
                  {cluster.count}
                </span>
              </button>
            ))}
            {selectedCluster !== null && (
              <button
                onClick={() => setSelectedCluster(null)}
                className="px-3 py-1 rounded-full bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200"
              >
                Show Allddd
              </button>
            )}
          </div>
          
          {/* Selected cluster details */}
          {selectedCluster !== null && (
            <div className="mt-4 p-4 bg-gray-100 dark:bg-gray-700 rounded-lg">
              <h4 className="font-medium">Cluster {selectedCluster} Details</h4>
              <p className="text-sm mt-1">
                {analysisData.clusters.find(c => c.cluster_id === selectedCluster)?.items.length || 0} holds detected
              </p>
              <p className="text-sm mt-1">
                Average HSdfV: {analysisData.clusters.find(c => c.cluster_id === selectedCluster)?.avg_hsv.map(v => Math.round(v)).join(', ')}
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}