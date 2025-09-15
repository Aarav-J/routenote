"use client";

import { useEffect, useRef, useState } from "react";
import { AnalysisResult, BoundingBox, Cluster, drawBoundingBoxes, ensureHoldId, isDarkColor, isPointInBox } from "../utils/imageProcessing";
import HoldNoteModal from "./HoldNoteModal";

interface ClimbingImageProps {
  originalImage: string;
  analysisData: AnalysisResult | null;
  isLoading?: boolean;
  onSaveNotes?: (notes: Record<string, string>) => void;
}

export default function ClimbingImageAnalyzer({ 
  originalImage, 
  analysisData, 
  isLoading = false,
  onSaveNotes
}: ClimbingImageProps) {
  const [selectedCluster, setSelectedCluster] = useState<number | null>(null);
  const [selectedHold, setSelectedHold] = useState<string | null>(null);
  const [currentNote, setCurrentNote] = useState<string>('');
  const [holdNotes, setHoldNotes] = useState<Record<string, string>>({});
  const [imageLoaded, setImageLoaded] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  
  // Effect to draw bounding boxes when data or selected holds change
  useEffect(() => {
    if (imageLoaded && analysisData) {
      drawBoundingBoxes(
        canvasRef.current, 
        imageRef.current, 
        analysisData, 
        selectedCluster,
        selectedHold,
        true
      );
    }
  }, [analysisData, selectedCluster, selectedHold, imageLoaded, holdNotes]);
  
  const [showNoteModal, setShowNoteModal] = useState<boolean>(false);
  
  // Apply notes to the analysis data
  useEffect(() => {
    if (analysisData && Object.keys(holdNotes).length > 0) {
      // Update cluster items with notes
      analysisData.clusters.forEach(cluster => {
        cluster.items.forEach(item => {
          const holdId = ensureHoldId(item).id;
          if (holdId && holdNotes[holdId]) {
            item.note = holdNotes[holdId];
          }
        });
      });
    }
  }, [analysisData, holdNotes]);
  
  // Handle cluster selection toggle
  const handleClusterClick = (clusterId: number) => {
    setSelectedCluster(selectedCluster === clusterId ? null : clusterId);
    setSelectedHold(null); // Clear selected hold when changing clusters
  };
  
  // Handle canvas click to select a hold
  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!analysisData || !imageRef.current || !canvasRef.current) return;
    
    // Get click coordinates relative to the canvas
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    // Convert screen coordinates to original image coordinates
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;
    
    // Find if we clicked on a hold in the selected cluster
    let foundHold: BoundingBox | null = null;
    
    analysisData.clusters.forEach(cluster => {
      // Only check holds in the selected cluster, or check all if no cluster is selected
      if (selectedCluster === null || cluster.cluster_id === selectedCluster) {
        cluster.items.forEach(hold => {
          if (isPointInBox(x, y, hold.bbox)) {
            foundHold = hold;
          }
        });
      }
    });
    
    // Update selected hold
    if (foundHold) {
      const holdId = ensureHoldId(foundHold).id as string;
      
      if (selectedHold === holdId) {
        // If the hold is already selected, open the note modal
        setShowNoteModal(true);
      } else {
        // Otherwise, select the hold
        setSelectedHold(holdId);
        
        // Load existing note if there is one
        if (holdNotes[holdId]) {
          setCurrentNote(holdNotes[holdId]);
        } else {
          setCurrentNote('');
        }
        
        // Open note modal immediately if hold has a note
        if (holdNotes[holdId]) {
          setShowNoteModal(true);
        }
      }
    } else {
      setSelectedHold(null);
      setCurrentNote('');
    }
  };
  
  // Save note for the selected hold
  const saveNote = (noteText: string = currentNote) => {
    if (selectedHold) {
      const updatedNotes = { ...holdNotes };
      
      if (noteText.trim() === '') {
        // If the note is empty, remove it
        delete updatedNotes[selectedHold];
      } else {
        // Otherwise, save it
        updatedNotes[selectedHold] = noteText;
      }
      
      setHoldNotes(updatedNotes);
      setCurrentNote(noteText);
      
      // Call the parent's save handler if provided
      if (onSaveNotes) {
        onSaveNotes(updatedNotes);
      }
    }
  };

  return (
    <div className="space-y-8">
      <div>
        <h3 className="text-lg font-medium text-gray-700 dark:text-gray-300 mb-2">Analyzed Image</h3>
        <div ref={containerRef} className="relative w-full mx-auto" style={{ maxWidth: '100%', textAlign: 'center' }}>
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
            
            {/* Canvas overlay for bounding boxes - we remove pointer-events-none to allow clicks */}
            <canvas
              ref={canvasRef}
              className="absolute top-0 left-0 rounded-lg cursor-pointer"
              style={{ position: 'absolute' }}
              onClick={handleCanvasClick}
            />
          </div>
          
          {/* Note Modal */}
          <HoldNoteModal
            isOpen={showNoteModal}
            onClose={() => setShowNoteModal(false)}
            note={currentNote}
            onSave={saveNote}
            title={`Add Note for Hold in Cluster ${selectedHold ? 
              analysisData?.clusters.find(c => 
                c.items.some(item => ensureHoldId(item).id === selectedHold)
              )?.cluster_id ?? '?' : '?'}`}
          />
          
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
                Show All
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
                Average HSV: {analysisData.clusters.find(c => c.cluster_id === selectedCluster)?.avg_hsv.map(v => Math.round(v)).join(', ')}
              </p>
              
              <div className="mt-3 text-sm">
                <p className="font-medium mb-1">Instructions:</p>
                <p>Click on any hold to select it and add notes.</p>
                <p>Holds with notes are highlighted with a special indicator.</p>
              </div>
            </div>
          )}
          
          {/* Note taking interface for selected hold */}
          {selectedHold && (
            <div className="mt-4 p-4 bg-gray-50 dark:bg-gray-800 border border-blue-300 dark:border-blue-700 rounded-lg">
              <div className="flex justify-between items-center">
                <h4 className="font-medium text-blue-600 dark:text-blue-400">Selected Hold</h4>
                <button
                  onClick={() => setShowNoteModal(true)}
                  className="px-3 py-1 bg-blue-600 text-white rounded hover:bg-blue-700 flex items-center"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" />
                  </svg>
                  {currentNote ? "Edit Note" : "Add Note"}
                </button>
              </div>
              
              {currentNote && (
                <div className="mt-3 bg-white dark:bg-gray-700 p-3 rounded-md border border-gray-200 dark:border-gray-600">
                  <p className="text-sm whitespace-pre-wrap">{currentNote}</p>
                </div>
              )}
              
              <div className="mt-3 flex justify-end">
                <button 
                  onClick={() => setSelectedHold(null)}
                  className="px-3 py-1 bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded hover:bg-gray-300 dark:hover:bg-gray-600"
                >
                  Close
                </button>
              </div>
            </div>
          )}
          
          {/* Summary of all notes */}
          {Object.keys(holdNotes).length > 0 && (
            <div className="mt-4 p-4 bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-900/30 rounded-lg">
              <h4 className="font-medium text-yellow-800 dark:text-yellow-500">Hold Notes ({Object.keys(holdNotes).length})</h4>
              
              <div className="mt-2 max-h-60 overflow-y-auto">
                {Object.entries(holdNotes).map(([holdId, note]) => {
                  // Find the hold and its cluster
                  let holdCluster = -1;
                  let holdBox: number[] = [];
                  
                  analysisData?.clusters.forEach(cluster => {
                    cluster.items.forEach(item => {
                      if (ensureHoldId(item).id === holdId) {
                        holdCluster = cluster.cluster_id;
                        holdBox = item.bbox;
                      }
                    });
                  });
                  
                  return (
                    <div key={holdId} className="mb-2 p-2 bg-white dark:bg-gray-800 rounded border border-yellow-100 dark:border-yellow-900/20">
                      <div className="flex justify-between items-start">
                        <span className="font-medium text-sm">
                          Hold in Cluster {holdCluster}
                        </span>
                        <button 
                          onClick={() => {
                            // Select this hold
                            setSelectedCluster(holdCluster);
                            setSelectedHold(holdId);
                            setCurrentNote(note);
                          }}
                          className="text-xs px-2 py-1 bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400 rounded"
                        >
                          Edit
                        </button>
                      </div>
                      <p className="text-sm mt-1 whitespace-pre-wrap">{note}</p>
                    </div>
                  );
                })}
              </div>
              
              {Object.keys(holdNotes).length > 0 && (
                <div className="mt-3 flex justify-center">
                  <button 
                    onClick={() => {
                      // Export notes as JSON
                      const dataStr = JSON.stringify(holdNotes, null, 2);
                      const dataUri = `data:application/json;charset=utf-8,${encodeURIComponent(dataStr)}`;
                      
                      const linkElement = document.createElement('a');
                      linkElement.setAttribute('href', dataUri);
                      linkElement.setAttribute('download', 'climbing_wall_notes.json');
                      document.body.appendChild(linkElement);
                      linkElement.click();
                      document.body.removeChild(linkElement);
                    }}
                    className="px-3 py-1 bg-green-600 text-white rounded hover:bg-green-700"
                  >
                    Export Notes
                  </button>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}