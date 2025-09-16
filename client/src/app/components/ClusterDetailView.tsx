"use client";

import { useEffect, useRef, useState } from "react";
import { AnalysisResult, BoundingBox, Cluster, drawBoundingBoxes, ensureHoldId, isDarkColor, isPointInBox } from "../utils/imageProcessing";
import HoldNoteModal from "./HoldNoteModal";

interface ClusterDetailViewProps {
  originalImage: string;
  analysisData: AnalysisResult;
  clusterId: number;
  onClose: () => void;
  onSaveNotes: (notes: Record<string, string>) => void;
  existingNotes: Record<string, string>;
}

export default function ClusterDetailView({ 
  originalImage, 
  analysisData, 
  clusterId,
  onClose,
  onSaveNotes,
  existingNotes = {}
}: ClusterDetailViewProps) {
  const [selectedHold, setSelectedHold] = useState<string | null>(null);
  const [currentNote, setCurrentNote] = useState<string>('');
  const [holdNotes, setHoldNotes] = useState<Record<string, string>>(existingNotes);
  const [imageLoaded, setImageLoaded] = useState(false);
  const [showNoteModal, setShowNoteModal] = useState<boolean>(false);
  
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  
  // Get the current cluster data
  const currentCluster = analysisData.clusters.find(c => c.cluster_id === clusterId);
  
  // Load the hold notes from the existing notes
  useEffect(() => {
    setHoldNotes(existingNotes);
  }, [existingNotes]);

  // Effect to draw bounding boxes when data or selected holds change
  useEffect(() => {
    if (imageLoaded && analysisData) {
      drawBoundingBoxes(
        canvasRef.current, 
        imageRef.current, 
        analysisData, 
        clusterId,
        selectedHold,
        true
      );
    }
  }, [analysisData, clusterId, selectedHold, imageLoaded, holdNotes]);
  
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
    
    if (currentCluster) {
      currentCluster.items.forEach(hold => {
        if (isPointInBox(x, y, hold.bbox)) {
          foundHold = hold;
        }
      });
    }
    
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
      
      // Call the parent's save handler
      onSaveNotes(updatedNotes);
    }
  };

  // Get cluster-specific holds with notes
  const clusterHoldNotes = Object.entries(holdNotes).filter(([holdId]) => {
    let isInCluster = false;
    if (currentCluster) {
      currentCluster.items.forEach(item => {
        if (ensureHoldId(item).id === holdId) {
          isInCluster = true;
        }
      });
    }
    return isInCluster;
  });

  if (!currentCluster) {
    return <div>Cluster not found</div>;
  }

  const [r, g, b] = currentCluster.rgb_color;
  const clusterColorStyle = `rgb(${r}, ${g}, ${b})`;
  
  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg overflow-hidden max-h-[90vh] flex flex-col">
      {/* Header with cluster info and close button */}
      <div className="p-4 flex justify-between items-center border-b border-gray-200 dark:border-gray-700" 
           style={{ backgroundColor: clusterColorStyle, color: isDarkColor(currentCluster.rgb_color) ? 'white' : 'black' }}>
        <div>
          <h2 className="text-xl font-bold">Cluster {clusterId}</h2>
          <p className="text-sm opacity-90">
            {currentCluster.count} holds
            {clusterHoldNotes.length > 0 && ` • ${clusterHoldNotes.length} with notes`}
          </p>
        </div>
        <button 
          onClick={onClose}
          className="p-2 rounded-full bg-white bg-opacity-20 hover:bg-opacity-30 transition-colors"
          aria-label="Close"
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      {/* Two-column layout */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 p-6">
        {/* Left column - Image with holds */}
        <div className="flex flex-col">
          <h3 className="text-lg font-medium text-gray-700 dark:text-gray-300 mb-3">
            Cluster View
            <span className="ml-2 text-xs px-2 py-1 bg-gray-200 dark:bg-gray-700 rounded-full">
              {currentCluster.count} holds
            </span>
          </h3>
          
          <div className="relative w-full border rounded-lg overflow-hidden bg-gray-100 dark:bg-gray-900">
            {/* The main image */}
            <div style={{ position: 'relative', display: 'inline-block', width: '100%' }}>
              <img 
                ref={imageRef}
                src={originalImage}
                alt={`Climbing wall cluster ${clusterId}`}
                className="w-full h-auto max-h-[70vh] object-contain"
                style={{ display: 'block' }}
                onLoad={() => setImageLoaded(true)}
              />
              
              {/* Canvas overlay for bounding boxes */}
              <canvas
                ref={canvasRef}
                className="absolute top-0 left-0 cursor-pointer w-full h-full"
                onClick={handleCanvasClick}
              />
            </div>
          </div>
          
          <div className="mt-3 text-sm text-gray-500 dark:text-gray-400">
            Click on any hold to select it. Click again to edit notes.
          </div>
          
          {/* Mobile-only buttons for better UX on small screens */}
          <div className="flex justify-center mt-4 lg:hidden">
            <button
              onClick={() => selectedHold && setShowNoteModal(true)}
              disabled={!selectedHold}
              className={`px-4 py-2 rounded-lg flex items-center justify-center
                        ${!selectedHold 
                          ? 'bg-gray-300 dark:bg-gray-700 text-gray-500 dark:text-gray-400 cursor-not-allowed' 
                          : 'bg-blue-600 text-white hover:bg-blue-700'}`}
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" />
              </svg>
              {selectedHold && currentNote ? "Edit Selected Note" : "Add Note to Selected Hold"}
            </button>
          </div>
        </div>

        {/* Right column - Notes interface */}
        <div className="flex flex-col">
          <h3 className="text-lg font-medium text-gray-700 dark:text-gray-300 mb-3">
            Hold Notes
            <span className="ml-2 text-sm text-gray-500 dark:text-gray-400">
              ({clusterHoldNotes.length} / {currentCluster.count})
            </span>
          </h3>

          {/* Selected hold info */}
          {selectedHold && (
            <div className="mb-4 p-4 bg-gray-50 dark:bg-gray-800 border border-blue-300 dark:border-blue-700 rounded-lg">
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
              
              {currentNote ? (
                <div className="mt-3 bg-white dark:bg-gray-700 p-3 rounded-md border border-gray-200 dark:border-gray-600">
                  <p className="text-sm whitespace-pre-wrap">{currentNote}</p>
                </div>
              ) : (
                <div className="mt-3 text-sm text-gray-500 italic">
                  No notes yet. Click the button above to add a note.
                </div>
              )}
            </div>
          )}

          {/* List of notes for this cluster */}
          <div className="flex-grow overflow-y-auto bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg">
            {clusterHoldNotes.length > 0 ? (
              <div className="divide-y divide-gray-200 dark:divide-gray-700">
                {clusterHoldNotes.map(([holdId, note]) => (
                  <div 
                    key={holdId} 
                    className={`p-4 hover:bg-gray-50 dark:hover:bg-gray-700 cursor-pointer transition-colors
                              ${selectedHold === holdId ? 'bg-blue-50 dark:bg-blue-900/30' : ''}`}
                    onClick={() => {
                      setSelectedHold(holdId);
                      setCurrentNote(note);
                    }}
                  >
                    <div className="flex justify-between items-start">
                      <div className="font-medium text-sm mb-1">
                        Hold #{currentCluster.items.findIndex(
                          item => ensureHoldId(item).id === holdId
                        ) + 1}
                      </div>
                      <button 
                        onClick={(e) => {
                          e.stopPropagation();
                          setSelectedHold(holdId);
                          setCurrentNote(note);
                          setShowNoteModal(true);
                        }}
                        className="text-xs px-2 py-1 bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400 rounded"
                      >
                        Edit
                      </button>
                    </div>
                    <p className="text-sm text-gray-600 dark:text-gray-300 whitespace-pre-wrap line-clamp-2">
                      {note}
                    </p>
                  </div>
                ))}
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center h-40 text-gray-500 dark:text-gray-400">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12 mb-2 opacity-50" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                <p className="text-center">No notes added yet</p>
                <p className="text-center text-sm">Click on a hold to add notes</p>
              </div>
            )}
          </div>

          {/* Note Modal */}
          <HoldNoteModal
            isOpen={showNoteModal}
            onClose={() => setShowNoteModal(false)}
            note={currentNote}
            onSave={saveNote}
            title={`Add Note for Hold #${
              selectedHold 
                ? currentCluster.items.findIndex(
                    item => ensureHoldId(item).id === selectedHold
                  ) + 1
                : ''
            }`}
          />
        </div>
      </div>
    </div>
  );
}