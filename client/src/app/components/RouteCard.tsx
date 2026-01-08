"use client";

import Link from "next/link";
import { Route } from "../utils/routeTypes";

interface RouteCardProps {
  route: Route;
  onDelete?: (id: string) => void;
}

export default function RouteCard({ route, onDelete }: RouteCardProps) {
  const date = new Date(route.createdAt);
  const holdCount = route.analysisData?.clusters?.reduce((sum, c) => sum + c.count, 0) || 0;
  const noteCount = Object.keys(route.notes || {}).length;

  const handleDelete = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (onDelete && confirm(`Are you sure you want to delete "${route.name}"?`)) {
      onDelete(route.id);
    }
  };

  return (
    <Link href={`/routes/${route.id}`}>
      <div className="group relative overflow-hidden rounded-xl border border-[var(--border)] bg-[var(--card-background)] transition-all duration-300 hover:border-[var(--border-strong)] hover:shadow-[var(--shadow-primary)]">
        {/* Thumbnail */}
        <div className="relative aspect-video w-full overflow-hidden bg-[var(--background-raised)]">
          {route.thumbnail ? (
            <img
              src={route.thumbnail}
              alt={route.name}
              className="h-full w-full object-cover transition-transform duration-500 group-hover:scale-110"
            />
          ) : route.imageUrl ? (
            <img
              src={route.imageUrl}
              alt={route.name}
              className="h-full w-full object-cover transition-transform duration-500 group-hover:scale-110"
            />
          ) : (
            <div className="flex h-full items-center justify-center text-[var(--foreground-muted)]">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
            </div>
          )}
          
          {/* Gradient overlay */}
          <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/0 to-black/0 opacity-0 transition-opacity duration-300 group-hover:opacity-100" />
          
          {/* Stats overlay */}
          <div className="absolute bottom-2 left-2 right-2 flex gap-2 opacity-0 transition-opacity duration-300 group-hover:opacity-100">
            <div className="rounded-lg bg-black/60 px-2.5 py-1 text-xs font-medium text-white backdrop-blur-sm">
              {holdCount} holds
            </div>
            {noteCount > 0 && (
              <div className="rounded-lg bg-[var(--primary-soft)] px-2.5 py-1 text-xs font-medium text-[var(--primary-light)] backdrop-blur-sm">
                {noteCount} notes
              </div>
            )}
          </div>
        </div>

        {/* Content */}
        <div className="p-4">
          <div className="mb-2 flex items-start justify-between gap-2">
            <h3 className="line-clamp-1 flex-1 text-base font-semibold text-white group-hover:text-[var(--primary-light)] transition-colors">
              {route.name}
            </h3>
            {onDelete && (
              <button
                onClick={handleDelete}
                className="flex-shrink-0 rounded-lg p-1 text-[var(--foreground-muted)] opacity-0 transition-all hover:bg-red-500/20 hover:text-red-400 group-hover:opacity-100"
                aria-label="Delete route"
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                </svg>
              </button>
            )}
          </div>

          {route.description && (
            <p className="mb-3 line-clamp-2 text-sm text-[var(--foreground-muted)]">
              {route.description}
            </p>
          )}

          {/* Tags */}
          {route.tags && route.tags.length > 0 && (
            <div className="mb-3 flex flex-wrap gap-1.5">
              {route.tags.slice(0, 3).map((tag) => (
                <span
                  key={tag}
                  className="rounded-full border border-[var(--primary-soft)] bg-[var(--primary-softer)] px-2 py-0.5 text-xs font-medium text-[var(--primary-light)]"
                >
                  {tag}
                </span>
              ))}
              {route.tags.length > 3 && (
                <span className="rounded-full px-2 py-0.5 text-xs font-medium text-[var(--foreground-muted)]">
                  +{route.tags.length - 3}
                </span>
              )}
            </div>
          )}

          {/* Footer */}
          <div className="flex items-center justify-between text-xs text-[var(--foreground-subtle)]">
            <span>{date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}</span>
            <div className="flex items-center gap-3">
              <span className="flex items-center gap-1">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7h16M4 12h16M4 17h16" />
                </svg>
                {holdCount}
              </span>
              {noteCount > 0 && (
                <span className="flex items-center gap-1 text-[var(--primary-light)]">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                  </svg>
                  {noteCount}
                </span>
              )}
            </div>
          </div>
        </div>
      </div>
    </Link>
  );
}
