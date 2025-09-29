"use client";

/* eslint-disable @next/next/no-img-element */

import { useEffect, useRef, useState } from "react";
import { AnalysisResult, BoundingBox, drawBoundingBoxes, ensureHoldId, isDarkColor, isPointInBox } from "../utils/imageProcessing";
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
  const headerTextColor = isDarkColor(currentCluster.rgb_color) ? '#ffffff' : '#0f1820';
  const headerBackground = `linear-gradient(135deg, rgba(${r}, ${g}, ${b}, 0.95), rgba(${r}, ${g}, ${b}, 0.55))`;
  
  return (
    <div className="flex max-h-[90vh] flex-col overflow-hidden rounded-3xl border border-[var(--border)] bg-[color-mix(in_srgb,var(--background-raised)_92%,_black_8%)] shadow-[0_26px_80px_rgba(5,8,14,0.7)]">
      {/* Header with cluster info and close button */}
      <div
        className="flex items-center justify-between gap-4 border-b border-white/10 px-6 py-5"
        style={{ background: headerBackground, color: headerTextColor }}
      >
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.36em]">Cluster {clusterId}</p>
          <h2 className="mt-2 text-2xl font-semibold leading-tight">Color signature</h2>
          <p className="mt-2 text-sm opacity-85">
            {currentCluster.count} holds
            {clusterHoldNotes.length > 0 && ` • ${clusterHoldNotes.length} noted`}
          </p>
        </div>
        <button 
          onClick={onClose}
          className="inline-flex h-10 w-10 items-center justify-center rounded-full border border-white/30 bg-white/10 text-current transition hover:bg-white/20"
          aria-label="Close"
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      {/* Two-column layout */}
      <div className="grid grid-cols-1 gap-8 px-6 pb-6 pt-7 lg:grid-cols-2">
        {/* Left column - Image with holds */}
        <div className="flex flex-col">
          <div className="flex flex-wrap items-center gap-3">
            <h3 className="text-lg font-semibold text-white/90">Cluster view</h3>
            <span className="rounded-full border border-white/15 px-3 py-1 text-xs font-semibold uppercase tracking-[0.3em] text-[var(--foreground-muted)]">
              {currentCluster.count} holds
            </span>
          </div>
          
          <div className="relative mt-4 w-full overflow-hidden rounded-3xl border border-[var(--border)] bg-black/25">
            {/* The main image */}
            <div style={{ position: 'relative', display: 'inline-block', width: '100%' }}>
              <img 
                ref={imageRef}
                src={originalImage}
                alt={`Climbing wall cluster ${clusterId}`}
                className="w-full max-h-[70vh] rounded-3xl border border-white/10 object-contain"
                style={{ display: 'block' }}
                onLoad={() => setImageLoaded(true)}
              />
              
              {/* Canvas overlay for bounding boxes */}
              <canvas
                ref={canvasRef}
                className="absolute top-0 left-0 h-full w-full cursor-pointer rounded-3xl"
                onClick={handleCanvasClick}
              />
            </div>
          </div>
          
          <div className="mt-4 rounded-2xl border border-white/10 bg-black/30 p-4 text-sm text-[var(--foreground-muted)]">
            Click once to select a hold. Click again to open the note editor.
          </div>
          
          {/* Mobile-only buttons for better UX on small screens */}
          <div className="flex justify-center mt-4 lg:hidden">
            <button
              onClick={() => selectedHold && setShowNoteModal(true)}
              disabled={!selectedHold}
              className={`flex items-center justify-center rounded-full px-5 py-2 text-sm font-semibold transition
                        ${!selectedHold 
                          ? 'cursor-not-allowed border border-white/10 text-white/30' 
                          : 'border border-[var(--primary)]/50 bg-[var(--primary)] text-white shadow-[0_10px_24px_rgba(197,24,241,0.35)] hover:bg-[var(--primary-strong)]'}`}
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
          <div className="flex flex-wrap items-baseline gap-3">
            <h3 className="text-lg font-semibold text-white/90">Hold notes</h3>
            <span className="text-sm text-[var(--foreground-muted)]">
              {clusterHoldNotes.length} of {currentCluster.count} holds documented
            </span>
          </div>

          {/* Selected hold info */}
          {selectedHold && (
            <div className="mb-5 rounded-3xl border border-[var(--primary)]/35 bg-[var(--primary-soft)]/40 p-5">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <h4 className="text-xs font-semibold uppercase tracking-[0.38em] text-[var(--primary)]">Selected hold</h4>
                <button
                  onClick={() => setShowNoteModal(true)}
                  className="inline-flex items-center gap-2 rounded-full bg-[var(--primary)] px-4 py-2 text-sm font-semibold text-white shadow-[0_12px_28px_rgba(197,24,241,0.35)] transition hover:bg-[var(--primary-strong)]"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" />
                  </svg>
                  {currentNote ? "Edit note" : "Add note"}
                </button>
              </div>
              
              {currentNote ? (
                <div className="mt-4 rounded-2xl border border-white/15 bg-black/30 p-4 text-sm text-white/90">
                  <p className="whitespace-pre-wrap leading-relaxed">{currentNote}</p>
                </div>
              ) : (
                <div className="mt-4 text-sm italic text-[var(--foreground-muted)]">
                  No notes yet. Click the button above to capture beta or setting tips.
                </div>
              )}
            </div>
          )}

          {/* List of notes for this cluster */}
          <div className="flex-grow overflow-y-auto rounded-3xl border border-[var(--border)] bg-black/20">
            {clusterHoldNotes.length > 0 ? (
              <div className="divide-y divide-white/5">
                {clusterHoldNotes.map(([holdId, note]) => (
                  <div 
                    key={holdId} 
                    className={`cursor-pointer px-5 py-4 transition-colors hover:bg-white/5 ${
                      selectedHold === holdId ? 'bg-[var(--primary-soft)]/30' : ''
                    }`}
                    onClick={() => {
                      setSelectedHold(holdId);
                      setCurrentNote(note);
                    }}
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div className="text-sm font-semibold text-white">
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
                        className="inline-flex items-center gap-2 rounded-full border border-white/20 px-3 py-1 text-xs font-semibold text-white transition hover:border-[var(--primary)]/60 hover:text-[var(--primary)]"
                      >
                        Edit
                      </button>
                    </div>
                    <p className="mt-2 line-clamp-3 text-sm text-[var(--foreground-muted)] whitespace-pre-wrap">{note}</p>
                  </div>
                ))}
              </div>
            ) : (
              <div className="flex h-40 flex-col items-center justify-center text-[var(--foreground-muted)]">
                <svg xmlns="http://www.w3.org/2000/svg" className="mb-3 h-12 w-12 opacity-50" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                <p className="text-center text-sm">No notes added yet. Select a hold to start documenting.</p>
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
