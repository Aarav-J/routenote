"use client";

import { useState, useEffect } from "react";
import { useRouter, useParams } from "next/navigation";
import { Route } from "../../utils/routeTypes";
import { getRoute, deleteRoute, updateRoute, exportRoute } from "../../utils/storage";
import ClimbingImageAnalyzer from "../../components/ClimbingImageAnalyzer";
import RouteForm from "../../components/RouteForm";
import { RouteMetadata } from "../../utils/routeTypes";
import { AnalysisResult } from "../../utils/imageProcessing";

export default function RouteDetailPage() {
  const router = useRouter();
  const params = useParams();
  const routeId = params.id as string;

  const [route, setRoute] = useState<Route | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isEditing, setIsEditing] = useState(false);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);

  useEffect(() => {
    const loadedRoute = getRoute(routeId);
    if (loadedRoute) {
      setRoute(loadedRoute);
    }
    setIsLoading(false);
  }, [routeId]);

  const handleUpdateMetadata = async (metadata: RouteMetadata) => {
    if (!route) return;

    const updated = await updateRoute(route.id, metadata);
    if (updated) {
      setRoute(updated);
      setIsEditing(false);
    }
  };

  const handleSaveNotes = (notes: Record<string, string>) => {
    if (!route) return;
    updateRoute(route.id, { notes }).then((updated) => {
      if (updated) {
        setRoute(updated);
      }
    });
  };

  const handleDelete = () => {
    if (!route) return;
    if (deleteRoute(route.id)) {
      router.push('/routes');
    }
  };

  const handleExport = () => {
    if (!route) return;
    const json = exportRoute(route);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${route.name.replace(/[^a-z0-9]/gi, '_').toLowerCase()}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  if (isLoading) {
    return (
      <div className="min-h-screen pt-20 pb-12">
        <div className="mx-auto max-w-7xl px-6">
          <div className="flex items-center justify-center py-20">
            <div className="flex flex-col items-center gap-4">
              <svg className="h-12 w-12 animate-spin text-[var(--primary)]" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              <p className="text-sm text-[var(--foreground-muted)]">Loading route...</p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (!route) {
    return (
      <div className="min-h-screen pt-20 pb-12">
        <div className="mx-auto max-w-7xl px-6">
          <div className="flex min-h-[400px] flex-col items-center justify-center rounded-2xl border border-[var(--border)] bg-[var(--card-background)] p-12 text-center">
            <h2 className="mb-2 text-xl font-semibold text-white">Route not found</h2>
            <p className="mb-6 text-[var(--foreground-muted)]">
              The route you're looking for doesn't exist or has been deleted.
            </p>
            <button
              onClick={() => router.push('/routes')}
              className="rounded-lg bg-gradient-to-r from-[var(--primary)] to-[var(--primary-light)] px-6 py-3 text-sm font-semibold text-white shadow-[var(--shadow-primary)] transition hover:shadow-[var(--shadow-primary-strong)]"
            >
              Back to Routes
            </button>
          </div>
        </div>
      </div>
    );
  }

  const date = new Date(route.createdAt);
  const holdCount = route.analysisData?.clusters?.reduce((sum, c) => sum + c.count, 0) || 0;
  const noteCount = Object.keys(route.notes || {}).length;

  return (
    <div className="min-h-screen pt-20 pb-12">
      <div className="mx-auto max-w-7xl px-6">
        {/* Header */}
        <div className="mb-8">
          <button
            onClick={() => router.back()}
            className="mb-4 inline-flex items-center gap-2 text-sm font-medium text-[var(--foreground-muted)] transition hover:text-white"
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            </svg>
            Back
          </button>

          <div className="mb-4 flex flex-wrap items-start justify-between gap-4">
            <div className="flex-1">
              <h1 className="mb-2 text-3xl font-bold text-white">{route.name}</h1>
              {route.description && (
                <p className="mb-3 text-[var(--foreground-muted)]">{route.description}</p>
              )}
              <div className="flex flex-wrap items-center gap-4 text-sm text-[var(--foreground-subtle)]">
                <span>{date.toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' })}</span>
                <span className="flex items-center gap-1">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7h16M4 12h16M4 17h16" />
                  </svg>
                  {holdCount} holds
                </span>
                {noteCount > 0 && (
                  <span className="flex items-center gap-1 text-[var(--primary-light)]">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                    </svg>
                    {noteCount} notes
                  </span>
                )}
              </div>
              {route.tags && route.tags.length > 0 && (
                <div className="mt-3 flex flex-wrap gap-2">
                  {route.tags.map((tag) => (
                    <span
                      key={tag}
                      className="rounded-full border border-[var(--primary-soft)] bg-[var(--primary-softer)] px-3 py-1 text-xs font-medium text-[var(--primary-light)]"
                    >
                      {tag}
                    </span>
                  ))}
                </div>
              )}
            </div>

            {/* Actions */}
            <div className="flex flex-wrap items-center gap-2">
              <button
                onClick={() => setIsEditing(true)}
                className="inline-flex items-center gap-2 rounded-lg border border-[var(--border)] bg-[var(--card-background)] px-4 py-2 text-sm font-semibold text-white transition hover:border-[var(--primary)] hover:text-[var(--primary-light)]"
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                </svg>
                Edit
              </button>
              <button
                onClick={handleExport}
                className="inline-flex items-center gap-2 rounded-lg border border-[var(--border)] bg-[var(--card-background)] px-4 py-2 text-sm font-semibold text-white transition hover:border-[var(--primary)] hover:text-[var(--primary-light)]"
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                </svg>
                Export
              </button>
              <button
                onClick={() => setShowDeleteConfirm(true)}
                className="inline-flex items-center gap-2 rounded-lg border border-red-500/30 bg-red-500/10 px-4 py-2 text-sm font-semibold text-red-400 transition hover:border-red-500/50 hover:bg-red-500/20"
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                </svg>
                Delete
              </button>
            </div>
          </div>
        </div>

        {/* Analysis Display */}
        <div className="rounded-2xl border border-[var(--border)] bg-[var(--card-background)] p-6">
          <ClimbingImageAnalyzer
            originalImage={route.imageUrl}
            analysisData={route.analysisData}
            onSaveNotes={handleSaveNotes}
          />
        </div>

        {/* Edit Form Modal */}
        {isEditing && (
          <RouteForm
            isOpen={isEditing}
            onClose={() => setIsEditing(false)}
            onSubmit={handleUpdateMetadata}
            initialData={{
              name: route.name,
              description: route.description,
              tags: route.tags
            }}
          />
        )}

        {/* Delete Confirmation Modal */}
        {showDeleteConfirm && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 px-4 py-10 backdrop-blur-sm">
            <div className="w-full max-w-md rounded-2xl border border-[var(--border)] bg-[var(--card-background)] p-6 shadow-[var(--shadow-primary-strong)]">
              <h2 className="mb-2 text-xl font-bold text-white">Delete Route?</h2>
              <p className="mb-6 text-[var(--foreground-muted)]">
                Are you sure you want to delete "{route.name}"? This action cannot be undone.
              </p>
              <div className="flex justify-end gap-3">
                <button
                  onClick={() => setShowDeleteConfirm(false)}
                  className="rounded-lg border border-[var(--border)] bg-[var(--background-raised)] px-5 py-2.5 text-sm font-semibold text-white transition hover:border-[var(--primary)] hover:text-[var(--primary-light)]"
                >
                  Cancel
                </button>
                <button
                  onClick={handleDelete}
                  className="rounded-lg bg-red-600 px-5 py-2.5 text-sm font-semibold text-white shadow-lg transition hover:bg-red-700"
                >
                  Delete
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
