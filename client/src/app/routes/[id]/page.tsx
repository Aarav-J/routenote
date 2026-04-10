"use client";

import { useState, useEffect } from "react";
import { useRouter, useParams } from "next/navigation";
import { Route } from "../../utils/routeTypes";
import { getRoute, deleteRoute, updateRoute } from "../../utils/storage";
import RouteForm from "../../components/RouteForm";
import { RouteMetadata } from "../../utils/routeTypes";
import { getImage } from "../../utils/imageStore";
import WorkspaceViewer from "../../components/WorkspaceViewer";

export default function RouteDetailPage() {
  const router = useRouter();
  const params = useParams();
  const routeId = params.id as string;

  const [route, setRoute] = useState<Route | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isEditing, setIsEditing] = useState(false);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [resolvedImage, setResolvedImage] = useState<string | null>(null);

  useEffect(() => {
    const loadedRoute = getRoute(routeId);
    if (loadedRoute) {
      setRoute(loadedRoute);
    }
    setIsLoading(false);
  }, [routeId]);

  useEffect(() => {
    let cancelled = false;
    async function resolve() {
      if (!route) return;
      if (route.imageUrl?.startsWith("idb:")) {
        const key = route.imageKey || route.imageUrl.slice(4);
        try {
          const dataUrl = await getImage(key);
          if (!cancelled) setResolvedImage(dataUrl);
        } catch {
          if (!cancelled) setResolvedImage(null);
        }
      } else {
        setResolvedImage(route.imageUrl);
      }
    }
    resolve();
    return () => {
      cancelled = true;
    };
  }, [route]);

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

  const handleSaveGrades = (holdGrades: Record<string, string>) => {
    if (!route) return;
    updateRoute(route.id, { clusterGrades: holdGrades }).then((updated) => {
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

  // (Export UI can be re-added in the Workspace header if desired)

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
              The route you&apos;re looking for doesn&apos;t exist or has been deleted.
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

  return (
    <div className="min-h-screen bg-[#0e0e0f] pt-16">
      {resolvedImage ? (
        <WorkspaceViewer
          projectName={route.name}
          originalImage={resolvedImage}
          analysisData={route.analysisData}
          notes={route.notes || {}}
          onNotesChange={handleSaveNotes}
          clusterGrades={route.clusterGrades || {}}
          onClusterGradesChange={handleSaveGrades}
          onSaveRoute={() => setIsEditing(true)}
        />
      ) : (
        <div className="flex min-h-[calc(100vh-4rem)] items-center justify-center text-sm text-[var(--foreground-muted)]">
          Loading image…
        </div>
      )}

      {/* Edit Form Modal */}
      {isEditing && (
        <RouteForm
          isOpen={isEditing}
          onClose={() => setIsEditing(false)}
          onSubmit={handleUpdateMetadata}
          initialData={{
            name: route.name,
            description: route.description,
            tags: route.tags,
          }}
        />
      )}

      {/* Delete Confirmation Modal */}
      {showDeleteConfirm && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 px-4 py-10 backdrop-blur-sm">
          <div className="w-full max-w-md rounded-2xl border border-[var(--border)] bg-[var(--card-background)] p-6 shadow-[var(--shadow-primary-strong)]">
            <h2 className="mb-2 text-xl font-bold text-white">Delete Route?</h2>
            <p className="mb-6 text-[var(--foreground-muted)]">
              Are you sure you want to delete &quot;{route.name}&quot;? This action cannot be undone.
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
  );
}
