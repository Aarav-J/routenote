"use client";

import { useMemo, useState, useEffect, ChangeEvent } from "react";
import { useRouter } from "next/navigation";
import { Route, RouteFilters } from "../utils/routeTypes";
import { getRoutes, deleteRoute, filterRoutes, getRouteStats } from "../utils/storage";
import RouteCard from "../components/RouteCard";

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
  const activeProjects = useMemo(() => {
    // No explicit "project" model yet; treat each saved route as a project.
    return routes.length;
  }, [routes.length]);

  const handleSearchChange = (e: ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setFilters((prev) => ({ ...prev, search: value || undefined }));
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-[#0e0e0f] pb-12 pt-20 text-white">
        <div className="mx-auto max-w-7xl px-6 lg:pl-[17.5rem]">
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
    <div className="min-h-screen bg-[#0e0e0f] pb-12 pt-20 text-white">
      <aside className="fixed bottom-0 left-0 top-16 hidden w-64 flex-col gap-2 bg-[#131314] p-4 lg:flex">
        <div className="mb-6 px-2">
          <div className="mb-4 flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-[#c284ff]">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-black" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.5 3.5h5m-8 7.5h11M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2h-1m-12 0H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
            </div>
            <div>
              <h4 className="text-sm font-bold text-white">Intelligence</h4>
              <p className="text-[10px] uppercase tracking-wider text-[#adaaab]">AI Route Analysis</p>
            </div>
          </div>
          <button
            onClick={() => router.push("/")}
            className="w-full rounded-lg bg-[#cc97ff] py-2 text-sm font-bold text-black shadow-[0_4px_15px_rgba(204,151,255,0.2)] transition hover:shadow-[0_4px_25px_rgba(204,151,255,0.4)]"
          >
            Analyze Image
          </button>
        </div>
        <nav className="flex flex-col gap-1">
          {[
            { label: "Visualizer" },
            { label: "Grades" },
            { label: "Texture" },
            { label: "Coordinates" },
            { label: "Export" },
          ].map((item) => (
            <a
              key={item.label}
              href="#"
              className="group flex items-center gap-3 rounded-lg px-4 py-3 text-[#adaaab] transition-all hover:bg-[#1a191b] hover:text-white"
              onClick={(e) => e.preventDefault()}
            >
              <span className="h-2 w-2 rounded-full bg-[#484849] group-hover:bg-[#cc97ff]" />
              <span className="text-sm font-medium">{item.label}</span>
            </a>
          ))}
        </nav>
        <div className="mt-auto rounded-xl bg-[#1a191b] p-4">
          <p className="mb-2 text-[10px] font-bold uppercase tracking-[0.1em] text-[#adaaab]">
            Storage Usage
          </p>
          <div className="mb-2 h-1.5 w-full overflow-hidden rounded-full bg-[#262627]">
            <div className="h-full w-[65%] rounded-full bg-[#cc97ff] shadow-[0_0_8px_rgba(204,151,255,0.6)]" />
          </div>
          <div className="flex items-center justify-between text-[10px] text-[#adaaab]">
            <span>1.2GB / 2GB</span>
            <span className="text-[#cc97ff]">65%</span>
          </div>
        </div>
      </aside>

      <main className="mx-auto max-w-7xl px-6 py-8 lg:pl-[17.5rem]">
        <section className="mb-10 grid grid-cols-1 gap-4 md:grid-cols-3">
          <div className="group relative overflow-hidden rounded-xl bg-[#131314] p-6">
            <div className="absolute right-0 top-0 p-4 opacity-10 transition-opacity group-hover:opacity-20">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-14 w-14 text-[#cc97ff]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 4h16M4 12h16M4 20h16" />
              </svg>
            </div>
            <p className="mb-1 text-sm font-medium text-[#adaaab]">Total Routes</p>
            <h2 className="font-headline text-4xl font-bold tracking-tight text-white">{stats.total}</h2>
            <div className="mt-4 flex items-center gap-2">
              <span className="rounded-full bg-[#69f6b8]/10 px-2 py-1 text-xs text-[#69f6b8]">Saved locally</span>
            </div>
          </div>

          <div className="group relative overflow-hidden rounded-xl bg-[#131314] p-6">
            <div className="absolute right-0 top-0 p-4 opacity-10 transition-opacity group-hover:opacity-20">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-14 w-14 text-[#cc97ff]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 6v6l4 2" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <p className="mb-1 text-sm font-medium text-[#adaaab]">Total Holds Documented</p>
            <h2 className="font-headline text-4xl font-bold tracking-tight text-white">{stats.totalHolds}</h2>
            <div className="mt-4 flex items-center gap-2">
              <span className="rounded-full bg-[#cc97ff]/10 px-2 py-1 text-xs text-[#cc97ff]">98% Accuracy</span>
            </div>
          </div>

          <div className="group relative overflow-hidden rounded-xl bg-[#131314] p-6">
            <div className="absolute right-0 top-0 p-4 opacity-10 transition-opacity group-hover:opacity-20">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-14 w-14 text-[#ff95a0]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M5 13l4 4L19 7" />
              </svg>
            </div>
            <p className="mb-1 text-sm font-medium text-[#adaaab]">Active Projects</p>
            <h2 className="font-headline text-4xl font-bold tracking-tight text-white">{activeProjects}</h2>
            <div className="mt-4 flex items-center gap-2">
              <span className="rounded-full bg-[#ff95a0]/10 px-2 py-1 text-xs text-[#ff95a0]">High priority</span>
            </div>
          </div>
        </section>

        <section className="mb-8 flex flex-col items-center justify-between gap-4 md:flex-row">
          <div className="group relative w-full md:w-96">
            <div className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-[#adaaab] transition-colors group-focus-within:text-[#cc97ff]">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
            </div>
            <input
              type="text"
              value={filters.search || ""}
              onChange={handleSearchChange}
              placeholder="Search routes by name or tag..."
              className="w-full rounded-xl bg-[#201f21] py-3 pl-10 pr-4 text-sm text-white placeholder:text-[#adaaab] focus:outline-none focus:ring-1 focus:ring-[#cc97ff]/40"
            />
          </div>

          <div className="flex w-full items-center gap-3 md:w-auto">
            <button
              type="button"
              className="inline-flex items-center gap-2 rounded-lg border border-[#484849]/40 bg-[#1a191b] px-4 py-2 text-sm font-medium text-[#adaaab] transition-all hover:text-white"
              onClick={() => setFilters((prev) => ({ ...prev, sortBy: prev.sortBy === "name" ? "newest" : "name" }))}
            >
              <span className="text-xs">≡</span>
              Grade
            </button>
            <button
              type="button"
              className="inline-flex items-center gap-2 rounded-lg border border-[#484849]/40 bg-[#1a191b] px-4 py-2 text-sm font-medium text-[#adaaab] transition-all hover:text-white"
              onClick={() => setFilters((prev) => ({ ...prev, sortBy: prev.sortBy === "oldest" ? "newest" : "oldest" }))}
            >
              <span className="text-xs">◼</span>
              Color
            </button>
            <button
              type="button"
              className="ml-auto inline-flex items-center gap-2 rounded-lg border border-[#cc97ff]/30 bg-[#cc97ff]/10 px-4 py-2 text-sm font-bold text-[#cc97ff] transition-all hover:bg-[#cc97ff]/20 md:ml-0"
              onClick={() => router.push("/")}
            >
              <span className="text-base leading-none">+</span>
              New Route
            </button>
          </div>
        </section>

        {filteredRoutes.length > 0 ? (
          <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
            {filteredRoutes.map((route) => (
              <RouteCard key={route.id} route={route} onDelete={handleDelete} />
            ))}
            <button
              type="button"
              onClick={() => router.push("/")}
              className="group flex min-h-[330px] flex-col items-center justify-center rounded-xl border-2 border-dashed border-[#484849]/40 p-8 transition-all hover:border-[#cc97ff]/50 hover:bg-[#cc97ff]/5"
            >
              <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-[#201f21] text-[#adaaab] transition-all group-hover:scale-110 group-hover:text-[#cc97ff]">
                <span className="text-2xl leading-none">+</span>
              </div>
              <p className="font-headline font-bold text-[#adaaab] transition-colors group-hover:text-white">
                Analyze New Wall
              </p>
              <p className="mt-2 text-center text-xs text-[#adaaab]">Upload a photo to start AI route detection</p>
            </button>
          </div>
        ) : routes.length === 0 ? (
          <div className="flex min-h-[400px] flex-col items-center justify-center rounded-2xl border border-[#262627] bg-[#131314] p-12 text-center">
            <h2 className="mb-2 text-xl font-semibold text-white">No routes yet</h2>
            <p className="mb-6 text-[#adaaab]">Analyze your first climbing wall to get started</p>
            <button
              onClick={() => router.push("/")}
              className="rounded-lg bg-gradient-to-r from-[#cc97ff] to-[#9c48ea] px-6 py-3 text-sm font-semibold text-black"
            >
              Analyze Image
            </button>
          </div>
        ) : (
          <div className="flex min-h-[400px] flex-col items-center justify-center rounded-2xl border border-[#262627] bg-[#131314] p-12 text-center">
            <h2 className="mb-2 text-xl font-semibold text-white">No routes match your filters</h2>
            <p className="mb-6 text-[#adaaab]">Try adjusting your search</p>
            <button
              onClick={() => setFilters({ sortBy: "newest" })}
              className="rounded-lg border border-[#484849] bg-[#1a191b] px-6 py-3 text-sm font-semibold text-white transition hover:border-[#cc97ff]"
            >
              Clear Filters
            </button>
          </div>
        )}
      </main>
    </div>
  );
}
