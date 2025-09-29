"use client";

/* eslint-disable @next/next/no-img-element */

import { useState, useRef, ChangeEvent } from "react";
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
    <div className="relative min-h-screen overflow-hidden">
      <div className="pointer-events-none absolute -top-48 left-1/2 h-[420px] w-[620px] -translate-x-1/2 rounded-full bg-[radial-gradient(circle_at_center,_rgba(197,24,241,0.22)_0%,_rgba(15,24,32,0.15)_65%,_rgba(15,24,32,0)_100%)] blur-3xl" />

      <main className="relative mx-auto flex min-h-screen max-w-6xl flex-col px-6 py-14 sm:px-10 lg:py-20">
        <header className="mx-auto max-w-3xl text-center">
          <div className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-4 py-1 text-xs font-medium uppercase tracking-[0.28em] text-[var(--foreground-muted)]">
            <span className="h-2 w-2 rounded-full bg-[var(--primary)] shadow-[0_0_12px_rgba(197,24,241,0.7)]" />
            Route Intelligence
          </div>
          <h1 className="mt-6 text-4xl font-semibold leading-tight tracking-tight text-white sm:text-5xl">
            Transform raw wall photos into actionable climbing insights
          </h1>
          <p className="mt-4 text-lg text-[var(--foreground-muted)]">
            Upload a climbing wall image, cluster holds by color, and capture precise notes for setting new problems or refining beta.
          </p>
        </header>

        <section className="mt-14 grid grid-cols-1 gap-8 lg:grid-cols-[minmax(0,360px)_1fr] lg:items-start">
          <div className="rounded-3xl border border-[var(--border)] bg-[color-mix(in_srgb,var(--background-raised)_90%,_black_10%)]/95 p-8 shadow-[var(--shadow)] backdrop-blur-xl">
            <div className="flex items-start justify-between">
              <div>
                <h2 className="text-xl font-semibold text-white">Upload a climbing wall</h2>
                <p className="mt-2 text-sm text-[var(--foreground-muted)]">
                  We support JPG, PNG, or WEBP images up to 10 MB.
                </p>
              </div>
              {selectedImage && (
                <span className="rounded-full bg-[var(--primary-soft)] px-3 py-1 text-xs font-medium text-[var(--primary)]">
                  Ready
                </span>
              )}
            </div>

            <div className="mt-6">
              <div
                className="group relative flex min-h-[220px] w-full cursor-pointer flex-col items-center justify-center overflow-hidden rounded-2xl border border-dashed border-white/10 bg-[var(--background-raised-soft)]/70 px-6 py-10 text-center transition-all duration-300 hover:border-[var(--primary)]/55 hover:bg-[var(--background-raised-soft)]"
                onClick={() => fileInputRef.current?.click()}
              >
                {selectedImage ? (
                  <div className="relative w-full max-w-md">
                    <div className="absolute inset-0 rounded-2xl bg-black/30 opacity-0 transition-opacity duration-300 group-hover:opacity-100" />
                    <img
                      src={selectedImage}
                      alt="Selected climbing wall"
                      className="relative z-10 w-full rounded-2xl border border-white/5 object-cover shadow-[0_18px_45px_rgba(5,8,16,0.45)]"
                    />
                    <div className="absolute inset-x-0 bottom-0 z-20 flex items-center justify-between rounded-b-2xl bg-black/45 px-4 py-2 text-xs text-white">
                      <span className="font-medium uppercase tracking-wide">Preview</span>
                      <span className="text-[var(--foreground-muted)]">Click to replace</span>
                    </div>
                  </div>
                ) : (
                  <div className="space-y-4">
                    <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-2xl bg-[var(--primary-soft)] text-[var(--primary)] shadow-[0_0_0_1px_rgba(197,24,241,0.1)]">
                      <svg
                        className="h-9 w-9"
                        xmlns="http://www.w3.org/2000/svg"
                        fill="none"
                        viewBox="0 0 24 24"
                        stroke="currentColor"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={1.5}
                          d="M4 16.5l4.2-4.2a2 2 0 012.828 0L16 17.5m-1.8-2.8l1.7-1.7a2 2 0 012.828 0L20 15m-7.5-6.5h.01M6 19.5h12a2 2 0 002-2V6.5a2 2 0 00-2-2H6a2 2 0 00-2 2v11a2 2 0 002 2z"
                        />
                      </svg>
                    </div>
                    <div className="space-y-1">
                      <p className="text-base font-medium text-white">Click to select or drop a file</p>
                      <p className="text-sm text-[var(--foreground-muted)]">We&apos;ll render a crisp preview and keep your file local.</p>
                    </div>
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
                <p className="mt-3 text-sm font-medium text-rose-400">{error}</p>
              )}

              <button
                onClick={handleUpload}
                disabled={!selectedImage || isLoading}
                className="group relative mt-6 inline-flex w-full items-center justify-center gap-2 rounded-full bg-[var(--primary)] px-6 py-3 text-sm font-semibold text-white shadow-[0_18px_45px_rgba(197,24,241,0.35)] transition duration-300 hover:bg-[var(--primary-strong)] disabled:pointer-events-none disabled:bg-white/8 disabled:text-white/35 disabled:shadow-none"
              >
                {isLoading ? (
                  <span className="flex items-center gap-2">
                    <svg
                      className="h-5 w-5 animate-spin text-white/90"
                      xmlns="http://www.w3.org/2000/svg"
                      fill="none"
                      viewBox="0 0 24 24"
                    >
                      <circle className="opacity-30" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                      <path className="opacity-80" fill="currentColor" d="M4 12a8 8 0 018-8V1.5C5.2 1.5 1.5 5.2 1.5 12H4z" />
                    </svg>
                    Processing image…
                  </span>
                ) : (
                  <>
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      className="h-5 w-5"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M3 7h18M7 3h10a4 4 0 014 4v10a4 4 0 01-4 4H7a4 4 0 01-4-4V7a4 4 0 014-4zm5 8h.01M12 16h.01m3-5h.01m-6 0h.01" />
                    </svg>
                    Analyze Image
                  </>
                )}
              </button>
            </div>
          </div>

          {analysisData && imageObjectUrl ? (
            <div className="rounded-3xl border border-[var(--border)] bg-[color-mix(in_srgb,var(--background-raised)_92%,_black_8%)]/95 p-8 shadow-[var(--shadow)] backdrop-blur-xl">
              <div className="flex flex-wrap items-start justify-between gap-4">
                <div>
                  <h2 className="text-2xl font-semibold text-white">Analysis results</h2>
                  <p className="mt-2 text-sm text-[var(--foreground-muted)]">
                    Explore detected clusters, annotate individual holds, and export your notes.
                  </p>
                </div>
                <span className="rounded-full bg-white/5 px-4 py-1 text-xs font-medium uppercase tracking-[0.28em] text-[var(--foreground-muted)]">
                  Beta Lab
                </span>
              </div>

              <div className="mt-8 space-y-10">
                <ClimbingImageAnalyzer
                  originalImage={imageObjectUrl}
                  analysisData={analysisData}
                  isLoading={isLoading}
                  onSaveNotes={handleSaveNotes}
                />

                {legendImage && (
                  <div className="rounded-2xl border border-[var(--border)] bg-black/20 p-5">
                    <div className="flex flex-wrap items-center justify-between gap-3">
                      <h3 className="text-lg font-medium text-white/90">Cluster legend</h3>
                      <span className="text-xs uppercase tracking-wide text-[var(--foreground-muted)]">
                        Auto-generated from your upload
                      </span>
                    </div>
                    <div className="relative mt-4 w-full overflow-hidden rounded-xl border border-white/5 bg-[var(--background-raised-soft)]">
                      <img
                        src={legendImage}
                        alt="Color cluster legend"
                        className="max-h-[420px] w-full rounded-xl object-contain"
                      />
                    </div>
                    <p className="mt-3 text-xs text-[var(--foreground-muted)]">
                      Each swatch maps to a detected color cluster with its hold count for quick reference while setting new problems.
                    </p>
                  </div>
                )}

                <div className="flex flex-wrap justify-center gap-3">
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
                    className="inline-flex items-center justify-center gap-2 rounded-full border border-white/20 px-5 py-2.5 text-sm font-semibold text-white transition hover:border-[var(--primary)]/50 hover:text-[var(--primary)]"
                  >
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      className="h-5 w-5"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4.5 19.5l3-3m0 0l-3-3m3 3H15a4.5 4.5 0 000-9h-1.5" />
                    </svg>
                    Analyze Another Image
                  </button>
                </div>
              </div>
            </div>
          ) : (
            <div className="hidden rounded-3xl border border-dashed border-white/10 bg-white/5 p-10 text-center text-sm text-[var(--foreground-muted)] lg:flex lg:flex-col lg:items-center lg:justify-center">
              <div className="rounded-full bg-[var(--primary-soft)] px-4 py-1 text-xs font-semibold text-[var(--primary)]">No analysis yet</div>
              <p className="mt-4 max-w-sm text-balance">Upload a wall photo to unlock clustering, dynamic overlays, and rich hold notes.</p>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}
