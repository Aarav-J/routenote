"use client";

import Link from "next/link";
import { Route } from "../utils/routeTypes";

interface RouteCardProps {
  route: Route;
  onDelete?: (id: string) => void;
}

function getRouteGrade(route: Route): string | null {
  const candidates = (route.tags || []).map((t) => t.trim());
  const found =
    candidates.find((t) => /^V\d{1,2}$/i.test(t)) ||
    candidates.find((t) => /^5\.\d{1,2}[abcd]?$/i.test(t));
  return found ? found.toUpperCase() : null;
}

function getRouteStatus(route: Route): { label: string; className: string } {
  const tags = (route.tags || []).map((t) => t.toLowerCase());
  if (tags.some((t) => t.includes("completed") || t === "done")) {
    return { label: "Completed", className: "bg-[#69f6b8] text-black" };
  }
  if (tags.some((t) => t.includes("review"))) {
    return { label: "In Review", className: "bg-[#ff909c] text-black" };
  }
  return { label: "Active", className: "bg-[#cc97ff] text-black" };
}

export default function RouteCard({ route, onDelete }: RouteCardProps) {
  const date = new Date(route.createdAt);
  const holdCount = route.analysisData?.clusters?.reduce((sum, c) => sum + c.count, 0) || 0;
  const noteCount = Object.keys(route.notes || {}).length;
  const grade = getRouteGrade(route) || "V?";
  const status = getRouteStatus(route);

  const handleDelete = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (onDelete && confirm(`Are you sure you want to delete "${route.name}"?`)) {
      onDelete(route.id);
    }
  };

  return (
    <Link href={`/routes/${route.id}`}>
      <div className="group overflow-hidden rounded-xl bg-[#1a191b] transition-all duration-300 hover:shadow-[0_0_40px_rgba(204,151,255,0.08)]">
        <div className="relative h-48 w-full overflow-hidden bg-[#262627]">
          {route.thumbnail ? (
            <img
              src={route.thumbnail}
              alt={route.name}
              className="h-full w-full object-cover transition-transform duration-500 group-hover:scale-105"
            />
          ) : route.imageUrl ? (
            <img
              src={route.imageUrl}
              alt={route.name}
              className="h-full w-full object-cover transition-transform duration-500 group-hover:scale-105"
            />
          ) : (
            <div className="flex h-full items-center justify-center text-[#adaaab]">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
            </div>
          )}

          <div className="absolute left-3 top-3">
            <span className={`rounded-full px-3 py-1 text-[10px] font-bold uppercase tracking-widest ${status.className}`}>
              {status.label}
            </span>
          </div>

          <div className="absolute bottom-3 right-3 rounded bg-black/60 px-2 py-1 text-xs text-white backdrop-blur-md">
            {grade}
          </div>
        </div>

        <div className="p-5">
          <div className="mb-2 flex items-start justify-between gap-3">
            <h3 className="font-headline line-clamp-1 text-lg font-bold leading-tight text-white transition-colors group-hover:text-[#cc97ff]">
              {route.name}
            </h3>
            {onDelete && (
              <button
                onClick={handleDelete}
                className="rounded-md p-1 text-[#adaaab] opacity-80 transition-colors hover:text-white"
                aria-label="More options"
              >
                <span className="text-lg leading-none">⋮</span>
              </button>
            )}
          </div>

          {route.description && (
            <p className="mb-3 line-clamp-2 text-sm text-[#adaaab]">
              {route.description}
            </p>
          )}

          <div className="mb-4 flex items-center gap-3 text-xs text-[#adaaab]">
            <span className="flex items-center gap-1">
              <span className="text-[12px]">🗓</span>
              {date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}
            </span>
            <span className="flex items-center gap-1">
              <span className="text-[12px]">●</span>
              {holdCount} holds
            </span>
            {noteCount > 0 && (
              <span className="flex items-center gap-1 text-[#cc97ff]">
                <span className="text-[12px]">✎</span>
                {noteCount} notes
              </span>
            )}
          </div>

          {route.tags && route.tags.length > 0 && (
            <div className="flex flex-wrap gap-2">
              {route.tags.slice(0, 3).map((tag) => (
                <span
                  key={tag}
                  className="rounded-full border border-[#484849]/30 bg-[#262627] px-2 py-0.5 text-[10px] text-[#adaaab]"
                >
                  {tag}
                </span>
              ))}
              {route.tags.length > 3 && (
                <span className="rounded-full px-2 py-0.5 text-[10px] font-medium text-[#adaaab]">
                  +{route.tags.length - 3}
                </span>
              )}
            </div>
          )}
        </div>
      </div>
    </Link>
  );
}
