"use client";

import { RouteFilters } from "../utils/routeTypes";

interface SearchAndFilterProps {
  filters: RouteFilters;
  onFiltersChange: (filters: RouteFilters) => void;
  totalRoutes: number;
  filteredCount: number;
}

export default function SearchAndFilter({
  filters,
  onFiltersChange,
  totalRoutes,
  filteredCount
}: SearchAndFilterProps) {
  return (
    <div className="space-y-4">
      {/* Search */}
      <div className="relative">
        <div className="absolute left-4 top-1/2 -translate-y-1/2 text-[var(--foreground-muted)]">
          <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
        </div>
        <input
          type="text"
          placeholder="Search routes by name, description, or tags..."
          value={filters.search || ''}
          onChange={(e) => onFiltersChange({ ...filters, search: e.target.value || undefined })}
          className="w-full rounded-lg border border-[var(--border)] bg-[var(--card-background)] pl-12 pr-4 py-3 text-white placeholder:text-[var(--foreground-subtle)] focus:border-[var(--primary)] focus:outline-none focus:ring-2 focus:ring-[var(--primary-soft)]"
        />
      </div>

      {/* Filter Bar */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        {/* Sort */}
        <div className="flex items-center gap-2">
          <label className="text-sm font-medium text-[var(--foreground-muted)]">Sort:</label>
          <select
            value={filters.sortBy || 'newest'}
            onChange={(e) => onFiltersChange({ ...filters, sortBy: e.target.value as RouteFilters['sortBy'] })}
            className="rounded-lg border border-[var(--border)] bg-[var(--card-background)] px-3 py-2 text-sm text-white focus:border-[var(--primary)] focus:outline-none focus:ring-2 focus:ring-[var(--primary-soft)]"
          >
            <option value="newest">Newest First</option>
            <option value="oldest">Oldest First</option>
            <option value="name">Name (A-Z)</option>
          </select>
        </div>

        {/* Results Count */}
        <div className="text-sm text-[var(--foreground-muted)]">
          {filteredCount === totalRoutes ? (
            <span>{totalRoutes} {totalRoutes === 1 ? 'route' : 'routes'}</span>
          ) : (
            <span>
              Showing {filteredCount} of {totalRoutes} {totalRoutes === 1 ? 'route' : 'routes'}
            </span>
          )}
        </div>

        {/* Clear Filters */}
        {(filters.search || filters.sortBy !== 'newest') && (
          <button
            onClick={() => onFiltersChange({ sortBy: 'newest' })}
            className="text-sm font-medium text-[var(--primary-light)] transition hover:text-[var(--primary)]"
          >
            Clear filters
          </button>
        )}
      </div>
    </div>
  );
}
