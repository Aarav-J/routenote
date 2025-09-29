"use client";

/* eslint-disable @next/next/no-img-element */

import { useEffect, useRef, useState } from "react";
import { AnalysisResult, BoundingBox, Cluster, drawBoundingBoxes, ensureHoldId, isDarkColor, isPointInBox } from "../utils/imageProcessing";
import HoldNoteModal from "./HoldNoteModal";
import ClusterDetailView from "./ClusterDetailView";

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
  
  // State for cluster detail view
  const [showClusterDetail, setShowClusterDetail] = useState<boolean>(false);
  const [detailClusterId, setDetailClusterId] = useState<number | null>(null);
  
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
    // Open cluster detail view instead of just selecting
    setDetailClusterId(clusterId);
    setShowClusterDetail(true);
    
    // Also update the selected cluster in the main view
    setSelectedCluster(clusterId);
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
    <div className="space-y-12">
      {/* Show cluster detail view when a cluster is selected for detail */}
      {showClusterDetail && detailClusterId !== null && analysisData && (
        <div className="fixed inset-0 z-50 flex items-center justify-center overflow-y-auto bg-black/70 px-4 py-10 backdrop-blur-sm">
          <div className="w-full max-w-6xl">
            <ClusterDetailView
              originalImage={originalImage}
              analysisData={analysisData}
              clusterId={detailClusterId}
              onClose={() => {
                setShowClusterDetail(false);
                // Redraw the main canvas after returning from detail view
                setTimeout(() => {
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
                }, 100);
              }}
              onSaveNotes={(updatedNotes) => {
                const newNotes = {...holdNotes, ...updatedNotes};
                setHoldNotes(newNotes);
                if (onSaveNotes) onSaveNotes(newNotes);
              }}
              existingNotes={holdNotes}
            />
          </div>
        </div>
      )}
    
      <div>
        <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
          <h3 className="text-lg font-semibold text-white/90">Analyzed image</h3>
          <span className="rounded-full border border-white/10 px-3 py-1 text-xs font-medium uppercase tracking-[0.28em] text-[var(--foreground-muted)]">
            Overlay View
          </span>
        </div>
        <div
          ref={containerRef}
          className="relative mx-auto w-full rounded-3xl border border-[var(--border)] bg-black/25 p-4 shadow-[inset_0_1px_0_rgba(255,255,255,0.05)]"
          style={{ maxWidth: '100%', textAlign: 'center' }}
        >
          {/* The main image */}
          <div style={{ position: 'relative', display: 'inline-block', maxWidth: '100%' }}>
            <img 
              ref={imageRef}
              src={originalImage}
              alt="Climbing wall with hold detection"
              className="max-h-[600px] max-w-full rounded-2xl border border-white/5"
              style={{ display: 'block' }}
              onLoad={() => setImageLoaded(true)}
            />
            
            {/* Canvas overlay for bounding boxes - we remove pointer-events-none to allow clicks */}
            <canvas
              ref={canvasRef}
              className="absolute top-0 left-0 cursor-pointer rounded-2xl"
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
            <div className="absolute inset-0 flex items-center justify-center rounded-2xl bg-black/60 backdrop-blur-sm">
              <div className="flex flex-col items-center text-white">
                <svg className="mb-3 h-10 w-10 animate-spin text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                <span className="text-sm text-[var(--foreground-muted)]">Analyzing image…</span>
              </div>
            </div>
          )}
        </div>
        <p className="mt-3 text-sm text-[var(--foreground-muted)]">
          Hover and click to inspect clusters. Each bounding box is tagged with its color group for precision.
        </p>
      </div>
      
      {/* Cluster selection buttons */}
      {analysisData && analysisData.clusters && analysisData.clusters.length > 0 && (
        <div className="mt-10">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <h3 className="text-lg font-semibold text-white/90">Color clusters</h3>
            <p className="text-sm text-[var(--foreground-muted)]">
              Tap a cluster to open a focused detail view and annotate individual holds.
            </p>
          </div>
          <div className="mt-4 flex flex-wrap gap-3">
            {analysisData.clusters.map((cluster: Cluster) => (
              <button
                key={cluster.cluster_id}
                onClick={() => handleClusterClick(cluster.cluster_id)}
                className={`group inline-flex items-center gap-2 rounded-full px-4 py-2 text-sm font-medium shadow-[0_12px_25px_rgba(5,8,14,0.35)] transition ${
                  selectedCluster === cluster.cluster_id
                    ? 'ring-2 ring-offset-2 ring-[var(--primary)] ring-offset-black/60'
                    : 'ring-1 ring-inset ring-white/15'
                }`}
                style={{
                  backgroundColor: `rgb(${cluster.rgb_color[0]}, ${cluster.rgb_color[1]}, ${cluster.rgb_color[2]})`,
                  color: isDarkColor(cluster.rgb_color) ? 'white' : 'black'
                }}
              >
                <span className="font-semibold tracking-wide">Cluster {cluster.cluster_id}</span>
                <span className="ml-1 rounded-full bg-black/20 px-2 text-xs font-semibold uppercase tracking-wide">
                  {cluster.count}
                </span>
                
                {/* Show note count if any holds in this cluster have notes */}
                {Object.entries(holdNotes).filter(([holdId]) => {
                  return cluster.items.some(item => ensureHoldId(item).id === holdId);
                }).length > 0 && (
                  <span className="ml-1 flex h-5 w-5 items-center justify-center rounded-full bg-white/30 text-black/70">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-3.5 w-3.5" viewBox="0 0 20 20" fill="currentColor">
                      <path d="M13.586 3.586a2 2 0 112.828 2.828l-.793.793-2.828-2.828.793-.793zM11.379 5.793L4 13.172V15h1.828l7.38-7.379-1.83-1.828z" />
                    </svg>
                  </span>
                )}
              </button>
            ))}
            {selectedCluster !== null && (
              <button
                onClick={() => setSelectedCluster(null)}
                className="inline-flex items-center justify-center gap-2 rounded-full border border-white/20 px-4 py-2 text-sm font-semibold text-white transition hover:border-[var(--primary)]/50 hover:text-[var(--primary)]"
              >
                Show All
              </button>
            )}
          </div>
          
          {/* Selected cluster details */}
          {selectedCluster !== null && (
            <div className="mt-5 rounded-3xl border border-[var(--border)] bg-white/5 p-5">
              <h4 className="text-xs font-semibold uppercase tracking-[0.32em] text-[var(--foreground-muted)]">
                Cluster {selectedCluster} Summary
              </h4>
              <p className="mt-2 text-base font-medium text-white">
                {analysisData.clusters.find(c => c.cluster_id === selectedCluster)?.items.length || 0} holds detected
              </p>
              <p className="mt-1 text-sm text-[var(--foreground-muted)]">
                Avg HSV: {analysisData.clusters.find(c => c.cluster_id === selectedCluster)?.avg_hsv.map(v => Math.round(v)).join(', ')}
              </p>
              
              <div className="mt-4 rounded-2xl border border-white/10 bg-black/30 p-4 text-sm text-[var(--foreground-muted)]">
                <p className="font-medium text-white">Tips</p>
                <p className="mt-2">• Click any hold to select it and capture context-specific beta.</p>
                <p className="mt-1">• Holds with notes display the ✎ indicator inside their cluster chip.</p>
              </div>
            </div>
          )}
          
          {/* Note taking interface for selected hold */}
          {selectedHold && (
            <div className="mt-6 rounded-3xl border border-[var(--primary)]/35 bg-[var(--primary-soft)]/45 p-5">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <h4 className="text-xs font-semibold uppercase tracking-[0.38em] text-[var(--primary)]">Selected Hold</h4>
                <button
                  onClick={() => setShowNoteModal(true)}
                  className="inline-flex items-center gap-2 rounded-full bg-[var(--primary)] px-4 py-2 text-sm font-semibold text-white shadow-[0_12px_28px_rgba(197,24,241,0.35)] transition hover:bg-[var(--primary-strong)]"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" />
                  </svg>
                  {currentNote ? "Edit Note" : "Add Note"}
                </button>
              </div>
              
              {currentNote && (
                <div className="mt-4 rounded-2xl border border-white/15 bg-black/30 p-4 text-sm text-white/90">
                  <p className="whitespace-pre-wrap leading-relaxed">{currentNote}</p>
                </div>
              )}
              
              <div className="mt-4 flex justify-end">
                <button 
                  onClick={() => setSelectedHold(null)}
                  className="inline-flex items-center gap-2 rounded-full border border-white/20 px-4 py-2 text-sm font-medium text-white transition hover:border-[var(--primary)]/60 hover:text-[var(--primary)]"
                >
                  Close
                </button>
              </div>
            </div>
          )}
          
          {/* Summary of all notes */}
          {Object.keys(holdNotes).length > 0 && (
            <div className="mt-8 rounded-3xl border border-[var(--border)] bg-white/5 p-5">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <h4 className="text-base font-semibold text-white">
                  Hold notes <span className="text-sm font-medium text-[var(--foreground-muted)]">({Object.keys(holdNotes).length})</span>
                </h4>
                <span className="rounded-full bg-black/30 px-3 py-1 text-xs font-medium uppercase tracking-[0.3em] text-[var(--foreground-muted)]">
                  Saved Locally
                </span>
              </div>
              
              <div className="mt-4 max-h-64 space-y-3 overflow-y-auto pr-1">
                {Object.entries(holdNotes).map(([holdId, note]) => {
                  // Find the hold and its cluster
                  let holdCluster = -1;
                  
                  analysisData?.clusters.forEach(cluster => {
                    cluster.items.forEach(item => {
                      if (ensureHoldId(item).id === holdId) {
                        holdCluster = cluster.cluster_id;
                      }
                    });
                  });
                  
                  return (
                    <div key={holdId} className="rounded-2xl border border-white/10 bg-black/35 p-4">
                      <div className="flex flex-wrap items-start justify-between gap-3">
                        <div>
                          <span className="text-xs font-semibold uppercase tracking-[0.28em] text-[var(--foreground-muted)]">
                            Cluster {holdCluster}
                          </span>
                          <p className="mt-1 text-sm font-medium text-white">Hold {holdId}</p>
                        </div>
                        <button 
                          onClick={() => {
                            // Select this hold
                            setSelectedCluster(holdCluster);
                            setSelectedHold(holdId);
                            setCurrentNote(note);
                          }}
                          className="inline-flex items-center gap-2 rounded-full border border-white/20 px-3 py-1 text-xs font-semibold text-white transition hover:border-[var(--primary)]/60 hover:text-[var(--primary)]"
                        >
                          Edit
                        </button>
                      </div>
                      <p className="mt-3 text-sm leading-relaxed text-[var(--foreground-muted)] whitespace-pre-wrap">{note}</p>
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
                    className="inline-flex items-center gap-2 rounded-full bg-[var(--accent)]/90 px-5 py-2 text-sm font-semibold text-black transition hover:bg-[var(--accent)]"
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
