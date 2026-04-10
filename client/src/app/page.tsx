"use client";

/* eslint-disable @next/next/no-img-element */

import { useState, useRef, ChangeEvent } from "react";
import { useRouter } from "next/navigation";
import ClimbingImageAnalyzer from "./components/ClimbingImageAnalyzer";
import RouteForm from "./components/RouteForm";
import { AnalysisResult } from "./utils/imageProcessing";
import { saveRoute } from "./utils/storage";
import { RouteMetadata } from "./utils/routeTypes";

export default function Home() {
  const router = useRouter();
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [imageObjectUrl, setImageObjectUrl] = useState<string | null>(null);
  const [originalFilename, setOriginalFilename] = useState<string>("");
  const [analysisData, setAnalysisData] = useState<AnalysisResult | null>(null);
  const [legendImage, setLegendImage] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [showSaveForm, setShowSaveForm] = useState<boolean>(false);
  const [notes, setNotes] = useState<Record<string, string>>({});
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

    setOriginalFilename(file.name);

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
      setNotes({});
      setShowSaveForm(false);
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

    // Auto-save and redirect to Routes (no scrolling on landing page)
    const defaultMetadata: RouteMetadata = {
      name: `Route ${new Date().toLocaleDateString()}`,
      description: "Auto-saved from analysis",
      tags: [],
    };

    const saved = await saveRoute(
      selectedImage, // persistable data URL (avoid ephemeral object URLs)
      originalFilename || defaultMetadata.name,
      data,
      {}, // notes start empty; user can add them in the route detail view
      defaultMetadata
    );

    router.push(`/routes/${saved.id}`);
    setIsLoading(false);
  } catch (err) {
    console.error('Error processing image:', err);
    setError('Failed to process the image. Please try again.');
    setIsLoading(false);
  }
};
  // Save notes to state
  const handleSaveNotes = (newNotes: Record<string, string>) => {
    setNotes(newNotes);
  };

  // Handle saving route with metadata
  const handleSaveRoute = async (metadata: RouteMetadata) => {
    if (!analysisData || !imageObjectUrl || !originalFilename) {
      setError('Missing data to save route');
      return;
    }

    try {
      const route = await saveRoute(
        imageObjectUrl,
        originalFilename,
        analysisData,
        notes,
        metadata
      );
      
      // Navigate to the routes page
      router.push(`/routes/${route.id}`);
    } catch (err) {
      console.error('Error saving route:', err);
      setError('Failed to save route. Please try again.');
    }
  };

  const resetAnalysisState = () => {
    setSelectedImage(null);
    setImageObjectUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return null;
    });
    setOriginalFilename("");
    setAnalysisData(null);
    setLegendImage(null);
    setNotes({});
    setShowSaveForm(false);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  return (
    <div className="min-h-screen bg-[#0e0e0f] pt-20 text-white">
      <main className="mx-auto max-w-7xl px-6 pb-24 pt-16">
        <section className="mb-16 text-center">
          <h1 className="mb-6 bg-gradient-to-b from-white to-[#adaaab] bg-clip-text text-5xl font-bold tracking-tight text-transparent md:text-6xl">
            Transform raw wall photos into <br className="hidden md:block" /> actionable climbing insights
          </h1>
          <p className="mx-auto max-w-2xl text-lg text-[#adaaab]">
            Precision route mapping and texture analysis powered by obsidian intelligence. High-fidelity data for the modern climber.
          </p>
        </section>

        <section className="mx-auto mb-16 max-w-4xl">
          <div className="group relative rounded-xl bg-[#131314] p-2 shadow-[0_0_60px_rgba(204,151,255,0.06)]">
            <div className="absolute -inset-1 rounded-xl bg-gradient-to-r from-[#cc97ff]/20 to-[#9c48ea]/20 opacity-20 blur-xl transition duration-500 group-hover:opacity-40" />
            <div
              className="relative flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed border-[#484849]/70 bg-[#1a191b] px-6 py-16 text-center transition-all duration-300 hover:border-[#cc97ff]/50 hover:shadow-[0_0_40px_rgba(204,151,255,0.15)] md:py-24"
              onClick={() => fileInputRef.current?.click()}
            >
              {selectedImage ? (
                <div className="w-full max-w-2xl space-y-6">
                  <img
                    src={selectedImage}
                    alt="Selected climbing wall"
                    className="mx-auto max-h-[460px] w-full rounded-lg border border-[#262627] object-contain"
                  />
                  <p className="text-sm text-[#adaaab]">Click anywhere in this panel to replace your image.</p>
                </div>
              ) : (
                <>
                  <div className="mb-6 flex h-16 w-16 items-center justify-center rounded-full bg-[#262627] text-[#cc97ff]">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-9 w-9" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16.5l4.2-4.2a2 2 0 012.828 0L16 17.5m-1.8-2.8l1.7-1.7a2 2 0 012.828 0L20 15m-7.5-6.5h.01M6 19.5h12a2 2 0 002-2V6.5a2 2 0 00-2-2H6a2 2 0 00-2 2v11a2 2 0 002 2z" />
                    </svg>
                  </div>
                  <h2 className="mb-2 text-2xl font-medium">Upload &amp; Analyze</h2>
                  <p className="mb-8 max-w-md text-[#adaaab]">
                    Drag your wall photo here or click to browse. Supports high-resolution JPG, PNG, and TIFF.
                  </p>
                </>
              )}
              <button
                type="button"
                className="rounded-full bg-gradient-to-r from-[#cc97ff] to-[#9c48ea] px-8 py-3 font-bold text-black transition-all hover:shadow-[0_0_20px_rgba(204,151,255,0.4)] active:scale-95"
                onClick={(e) => {
                  e.stopPropagation();
                  fileInputRef.current?.click();
                }}
              >
                Select Image
              </button>
              <input
                ref={fileInputRef}
                type="file"
                className="hidden"
                onChange={handleFileChange}
                accept="image/*"
              />
            </div>
          </div>
          {error && <p className="mt-4 text-sm font-medium text-rose-400">{error}</p>}
          <div className="mt-6 flex justify-center">
            <button
              onClick={handleUpload}
              disabled={!selectedImage || isLoading}
              className="inline-flex items-center justify-center rounded-full bg-[#262627] px-7 py-3 text-sm font-semibold text-white transition hover:bg-[#2c2c2d] disabled:cursor-not-allowed disabled:opacity-50"
            >
              {isLoading ? "Processing image..." : "Run Analysis"}
            </button>
          </div>
        </section>

        <section className="mb-12">
          <div className="mb-8 flex items-center justify-between">
            <h3 className="text-xl font-bold tracking-tight">Recent Analyses</h3>
            <button className="text-sm font-medium text-[#cc97ff] hover:underline">View all history</button>
          </div>
          <div className="grid grid-cols-1 gap-6 md:grid-cols-3 lg:grid-cols-4">
            <div className="overflow-hidden rounded-xl bg-[#131314] transition-colors hover:bg-[#1a191b]">
              <div className="aspect-[4/3] bg-[#262627]" />
              <div className="p-4">
                <div className="mb-1 flex items-start justify-between">
                  <h4 className="font-bold">The Overhang A2</h4>
                  <span className="text-xs text-[#adaaab]">V6</span>
                </div>
                <span className="text-xs text-[#adaaab]">2 hours ago</span>
              </div>
            </div>
            <div className="overflow-hidden rounded-xl bg-[#131314] transition-colors hover:bg-[#1a191b]">
              <div className="aspect-[4/3] bg-[#262627]" />
              <div className="p-4">
                <div className="mb-1 flex items-start justify-between">
                  <h4 className="font-bold">Limestone Peak</h4>
                  <span className="text-xs text-[#adaaab]">5.12a</span>
                </div>
                <span className="text-xs text-[#adaaab]">5 hours ago</span>
              </div>
            </div>
            <div className="overflow-hidden rounded-xl bg-[#131314] transition-colors hover:bg-[#1a191b]">
              <div className="aspect-[4/3] bg-[#262627]" />
              <div className="p-4">
                <div className="mb-1 flex items-start justify-between">
                  <h4 className="font-bold">Sector Delta</h4>
                  <span className="text-xs text-[#adaaab]">V4</span>
                </div>
                <span className="text-xs text-[#adaaab]">Yesterday</span>
              </div>
            </div>
            <div className="flex cursor-pointer flex-col items-center justify-center rounded-xl border border-dashed border-[#484849] bg-[#131314]/40 p-4 transition-all hover:border-[#cc97ff]/40">
              <div className="mb-2 flex h-10 w-10 items-center justify-center rounded-full border border-[#484849] text-[#adaaab]">+</div>
              <span className="text-xs font-medium tracking-tight text-[#adaaab]">New Project</span>
            </div>
          </div>
        </section>

        {analysisData && imageObjectUrl && (
          <section className="rounded-2xl border border-[#262627] bg-[#131314] p-8">
            <div className="flex flex-wrap items-start justify-between gap-4">
              <div>
                <h2 className="text-2xl font-semibold">Analysis results</h2>
                <p className="mt-2 text-sm text-[#adaaab]">
                  Explore detected clusters, annotate individual holds, and save your route.
                </p>
              </div>
              <div className="flex items-center gap-3">
                <button
                  onClick={() => setShowSaveForm(true)}
                  className="rounded-full bg-gradient-to-r from-[#cc97ff] to-[#9c48ea] px-5 py-2 text-sm font-semibold text-black"
                >
                  Save Route
                </button>
                <button
                  onClick={resetAnalysisState}
                  className="rounded-full border border-[#484849] px-5 py-2 text-sm font-semibold text-white hover:border-[#cc97ff]"
                >
                  Analyze Another
                </button>
              </div>
            </div>

            <div className="mt-8 space-y-10">
              <ClimbingImageAnalyzer
                originalImage={imageObjectUrl}
                analysisData={analysisData}
                isLoading={isLoading}
                onSaveNotes={handleSaveNotes}
              />

              {legendImage && (
                <div className="rounded-xl border border-[#262627] bg-black/20 p-5">
                  <h3 className="text-lg font-medium text-white/90">Cluster legend</h3>
                  <div className="relative mt-4 w-full overflow-hidden rounded-xl border border-white/5 bg-[#1a191b]">
                    <img
                      src={legendImage}
                      alt="Color cluster legend"
                      className="max-h-[420px] w-full rounded-xl object-contain"
                    />
                  </div>
                </div>
              )}
            </div>
          </section>
        )}
      </main>

      <footer className="border-t border-[#484849]/30 px-6 py-12">
        <div className="mx-auto flex max-w-7xl flex-col items-center justify-between gap-8 md:flex-row">
          <div className="text-lg font-bold tracking-tight text-[#cc97ff]">RouteNote</div>
          <div className="flex gap-8 text-sm text-[#adaaab]">
            <a className="transition-colors hover:text-[#cc97ff]" href="#">Documentation</a>
            <a className="transition-colors hover:text-[#cc97ff]" href="#">API</a>
            <a className="transition-colors hover:text-[#cc97ff]" href="#">Privacy</a>
            <a className="transition-colors hover:text-[#cc97ff]" href="#">Support</a>
          </div>
          <div className="text-[10px] uppercase tracking-widest text-[#adaaab]/50">© 2024 Obsidian Intelligence Systems</div>
        </div>
      </footer>

      {/* Save Route Form Modal */}
      {showSaveForm && (
        <RouteForm
          isOpen={showSaveForm}
          onClose={() => setShowSaveForm(false)}
          onSubmit={handleSaveRoute}
          defaultName={`Route ${new Date().toLocaleDateString()}`}
        />
      )}
    </div>
  );
}
