/**
 * localStorage utility functions for route management
 */

import { Route, RouteMetadata } from './routeTypes';
import { AnalysisResult } from './imageProcessing';
import { deleteImage, putImage } from "./imageStore";

const STORAGE_KEY = 'climbing_routes';
const NOTES_KEY = 'climbingWallNotes'; // Legacy key for backward compatibility

/**
 * Generate a unique route ID
 */
function generateRouteId(): string {
  return `route_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
}

/**
 * Create a thumbnail from an image URL (base64)
 */
async function createThumbnail(imageUrl: string, maxWidth: number = 300): Promise<string | undefined> {
  try {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    
    return new Promise((resolve) => {
      img.onload = () => {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        if (!ctx) {
          resolve(undefined);
          return;
        }
        
        // Calculate dimensions
        const ratio = Math.min(maxWidth / img.width, maxWidth / img.height);
        canvas.width = img.width * ratio;
        canvas.height = img.height * ratio;
        
        // Draw and convert to base64
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        const thumbnail = canvas.toDataURL('image/jpeg', 0.7);
        resolve(thumbnail);
      };
      
      img.onerror = () => resolve(undefined);
      img.src = imageUrl;
    });
  } catch (error) {
    console.error('Error creating thumbnail:', error);
    return undefined;
  }
}

/**
 * Get all saved routes
 */
export function getRoutes(): Route[] {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (!stored) return [];
    
    const routes = JSON.parse(stored) as Route[];
    return routes.sort((a, b) => b.createdAt - a.createdAt); // Sort by newest first
  } catch (error) {
    console.error('Error loading routes:', error);
    return [];
  }
}

/**
 * Get a single route by ID
 */
export function getRoute(id: string): Route | null {
  const routes = getRoutes();
  return routes.find(r => r.id === id) || null;
}

/**
 * Save a new route
 */
export async function saveRoute(
  imageUrl: string,
  originalFilename: string,
  analysisData: AnalysisResult,
  notes: Record<string, string>,
  metadata: RouteMetadata
): Promise<Route> {
  const routes = getRoutes();

  // Store large image payloads in IndexedDB to avoid localStorage quota limits.
  let storedImageUrl = imageUrl;
  let imageKey: string | undefined;
  try {
    if (imageUrl.startsWith("data:image/")) {
      imageKey = await putImage(imageUrl);
      storedImageUrl = `idb:${imageKey}`;
    }
  } catch (e) {
    // If IndexedDB is unavailable, fall back to whatever we were given.
    console.warn("IndexedDB image save failed; falling back to inline imageUrl", e);
  }
  
  const route: Route = {
    id: generateRouteId(),
    name: metadata.name || `Route ${new Date().toLocaleDateString()}`,
    description: metadata.description,
    tags: metadata.tags || [],
    createdAt: Date.now(),
    imageUrl: storedImageUrl,
    imageKey,
    originalFilename,
    analysisData,
    notes,
    clusterGrades: {},
    thumbnail: await createThumbnail(imageUrl)
  };
  
  routes.push(route);
  localStorage.setItem(STORAGE_KEY, JSON.stringify(routes));
  
  return route;
}

/**
 * Update an existing route
 */
export async function updateRoute(
  id: string,
  updates: Partial<RouteMetadata> & { notes?: Record<string, string>; clusterGrades?: Record<string, string> }
): Promise<Route | null> {
  const routes = getRoutes();
  const index = routes.findIndex(r => r.id === id);
  
  if (index === -1) return null;
  
  const route = routes[index];
  const updatedRoute: Route = {
    ...route,
    name: updates.name ?? route.name,
    description: updates.description ?? route.description,
    tags: updates.tags ?? route.tags,
    notes: updates.notes ?? route.notes,
    clusterGrades: updates.clusterGrades ?? route.clusterGrades
  };
  
  // Regenerate thumbnail if image URL changed
  if (updates.imageUrl && updates.imageUrl !== route.imageUrl) {
    let storedImageUrl = updates.imageUrl;
    let imageKey: string | undefined;
    try {
      if (updates.imageUrl.startsWith("data:image/")) {
        imageKey = await putImage(updates.imageUrl);
        storedImageUrl = `idb:${imageKey}`;
      }
    } catch (e) {
      console.warn("IndexedDB image update failed; falling back to inline imageUrl", e);
    }
    updatedRoute.imageUrl = storedImageUrl;
    updatedRoute.imageKey = imageKey;
    updatedRoute.thumbnail = await createThumbnail(updates.imageUrl);
  }
  
  routes[index] = updatedRoute;
  localStorage.setItem(STORAGE_KEY, JSON.stringify(routes));
  
  return updatedRoute;
}

/**
 * Delete a route
 */
export function deleteRoute(id: string): boolean {
  const routes = getRoutes();
  const routeToDelete = routes.find(r => r.id === id);
  const filtered = routes.filter(r => r.id !== id);
  
  if (filtered.length === routes.length) return false;
  
  localStorage.setItem(STORAGE_KEY, JSON.stringify(filtered));

  // Best-effort cleanup of IndexedDB image payload.
  const key = routeToDelete?.imageKey || (routeToDelete?.imageUrl?.startsWith("idb:") ? routeToDelete.imageUrl.slice(4) : undefined);
  if (key) {
    deleteImage(key).catch((e) => console.warn("Failed to delete stored image", e));
  }
  return true;
}

/**
 * Filter and sort routes
 */
export function filterRoutes(
  routes: Route[],
  filters: {
    search?: string;
    tags?: string[];
    dateFrom?: number;
    dateTo?: number;
    sortBy?: 'newest' | 'oldest' | 'name';
  }
): Route[] {
  let filtered = [...routes];
  
  // Search filter
  if (filters.search) {
    const searchLower = filters.search.toLowerCase();
    filtered = filtered.filter(route => 
      route.name.toLowerCase().includes(searchLower) ||
      route.description?.toLowerCase().includes(searchLower) ||
      route.tags.some(tag => tag.toLowerCase().includes(searchLower))
    );
  }
  
  // Tag filter
  if (filters.tags && filters.tags.length > 0) {
    filtered = filtered.filter(route =>
      filters.tags!.some(tag => route.tags.includes(tag))
    );
  }
  
  // Date range filter
  if (filters.dateFrom) {
    filtered = filtered.filter(route => route.createdAt >= filters.dateFrom!);
  }
  if (filters.dateTo) {
    filtered = filtered.filter(route => route.createdAt <= filters.dateTo!);
  }
  
  // Sort
  const sortBy = filters.sortBy || 'newest';
  filtered.sort((a, b) => {
    switch (sortBy) {
      case 'oldest':
        return a.createdAt - b.createdAt;
      case 'name':
        return a.name.localeCompare(b.name);
      case 'newest':
      default:
        return b.createdAt - a.createdAt;
    }
  });
  
  return filtered;
}

/**
 * Get route statistics
 */
export function getRouteStats(routes: Route[]): {
  total: number;
  totalHolds: number;
  totalNotes: number;
  oldestDate: number | null;
  newestDate: number | null;
} {
  if (routes.length === 0) {
    return {
      total: 0,
      totalHolds: 0,
      totalNotes: 0,
      oldestDate: null,
      newestDate: null
    };
  }
  
  const dates = routes.map(r => r.createdAt);
  const totalHolds = routes.reduce((sum, route) => 
    sum + (route.analysisData?.clusters?.reduce((s, c) => s + c.count, 0) || 0), 0
  );
  const totalNotes = routes.reduce((sum, route) => 
    sum + Object.keys(route.notes || {}).length, 0
  );
  
  return {
    total: routes.length,
    totalHolds,
    totalNotes,
    oldestDate: Math.min(...dates),
    newestDate: Math.max(...dates)
  };
}

/**
 * Export route as JSON
 */
export function exportRoute(route: Route): string {
  return JSON.stringify(route, null, 2);
}

/**
 * Import route from JSON (for future use)
 */
export function importRoute(json: string): Route | null {
  try {
    const route = JSON.parse(json) as Route;
    // Validate route structure
    if (route.id && route.name && route.analysisData) {
      return route;
    }
    return null;
  } catch (error) {
    console.error('Error importing route:', error);
    return null;
  }
}
