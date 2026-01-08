"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { Route, RouteFilters } from "../utils/routeTypes";
import { getRoutes, deleteRoute, filterRoutes, getRouteStats } from "../utils/storage";
import RouteCard from "../components/RouteCard";
import SearchAndFilter from "../components/SearchAndFilter";

export default function RoutesPage() {
  const router = useRouter();
  const [routes, setRoutes] = useState<Route[]>([]);
  const [filteredRoutes, setFilteredRoutes] = useState<Route[]>([]);
  const [filters, setFilters] = useState<RouteFilters>({ sortBy: 'newest' });
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const loadedRoutes = getRoutes();
    setRoutes(loadedRoutes);
    setIsLoading(false);
  }, []);

  useEffect(() => {
    const filtered = filterRoutes(routes, filters);
    setFilteredRoutes(filtered);
  }, [routes, filters]);

  const handleDelete = (id: string) => {
    if (deleteRoute(id)) {
      setRoutes(getRoutes());
    }
  };

  const stats = getRouteStats(routes);

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
              <p className="text-sm text-[var(--foreground-muted)]">Loading routes...</p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen pt-20 pb-12">
      <div className="mx-auto max-w-7xl px-6">
        {/* Header */}
        <div className="mb-8">
          <div className="mb-2 flex items-center gap-3">
            <h1 className="text-3xl font-bold text-white">My Routes</h1>
            <span className="rounded-full border border-[var(--border)] bg-[var(--primary-soft)] px-3 py-1 text-xs font-medium text-[var(--primary-light)]">
              {stats.total} {stats.total === 1 ? 'route' : 'routes'}
            </span>
          </div>
          <p className="text-[var(--foreground-muted)]">
            View and manage all your analyzed climbing routes
          </p>
        </div>

        {/* Stats Bar */}
        {stats.total > 0 && (
          <div className="mb-6 grid grid-cols-1 gap-4 sm:grid-cols-3">
            <div className="rounded-xl border border-[var(--border)] bg-[var(--card-background)] p-4">
              <div className="text-sm text-[var(--foreground-muted)]">Total Holds</div>
              <div className="mt-1 text-2xl font-bold text-white">{stats.totalHolds}</div>
            </div>
            <div className="rounded-xl border border-[var(--border)] bg-[var(--card-background)] p-4">
              <div className="text-sm text-[var(--foreground-muted)]">Total Notes</div>
              <div className="mt-1 text-2xl font-bold text-white">{stats.totalNotes}</div>
            </div>
            <div className="rounded-xl border border-[var(--border)] bg-[var(--card-background)] p-4">
              <div className="text-sm text-[var(--foreground-muted)]">Avg Holds/Route</div>
              <div className="mt-1 text-2xl font-bold text-white">
                {stats.total > 0 ? Math.round(stats.totalHolds / stats.total) : 0}
              </div>
            </div>
          </div>
        )}

        {/* Search and Filter */}
        {routes.length > 0 && (
          <div className="mb-6">
            <SearchAndFilter
              filters={filters}
              onFiltersChange={setFilters}
              totalRoutes={routes.length}
              filteredCount={filteredRoutes.length}
            />
          </div>
        )}

        {/* Routes Grid */}
        {filteredRoutes.length > 0 ? (
          <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3">
            {filteredRoutes.map((route) => (
              <RouteCard key={route.id} route={route} onDelete={handleDelete} />
            ))}
          </div>
        ) : routes.length === 0 ? (
          <div className="flex min-h-[400px] flex-col items-center justify-center rounded-2xl border border-[var(--border)] bg-[var(--card-background)] p-12 text-center">
            <div className="mb-4 rounded-full bg-[var(--primary-soft)] p-4">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12 text-[var(--primary-light)]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
              </svg>
            </div>
            <h2 className="mb-2 text-xl font-semibold text-white">No routes yet</h2>
            <p className="mb-6 text-[var(--foreground-muted)]">
              Analyze your first climbing wall to get started
            </p>
            <button
              onClick={() => router.push('/')}
              className="rounded-lg bg-gradient-to-r from-[var(--primary)] to-[var(--primary-light)] px-6 py-3 text-sm font-semibold text-white shadow-[var(--shadow-primary)] transition hover:shadow-[var(--shadow-primary-strong)]"
            >
              Analyze Image
            </button>
          </div>
        ) : (
          <div className="flex min-h-[400px] flex-col items-center justify-center rounded-2xl border border-[var(--border)] bg-[var(--card-background)] p-12 text-center">
            <div className="mb-4 rounded-full bg-[var(--primary-soft)] p-4">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12 text-[var(--primary-light)]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
            </div>
            <h2 className="mb-2 text-xl font-semibold text-white">No routes match your filters</h2>
            <p className="mb-6 text-[var(--foreground-muted)]">
              Try adjusting your search or filters
            </p>
            <button
              onClick={() => setFilters({ sortBy: 'newest' })}
              className="rounded-lg border border-[var(--border)] bg-[var(--card-background)] px-6 py-3 text-sm font-semibold text-white transition hover:border-[var(--primary)] hover:text-[var(--primary-light)]"
            >
              Clear Filters
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
