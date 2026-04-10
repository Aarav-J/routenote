/**
 * TypeScript interfaces for Route data model
 */

import { AnalysisResult } from './imageProcessing';

export interface Route {
  id: string;
  name: string;
  description?: string;
  tags: string[];
  createdAt: number;
  imageUrl: string;
  /**
   * If `imageUrl` starts with `idb:` then the remainder is the IndexedDB key.
   * This avoids blowing past localStorage quotas with large base64 payloads.
   */
  imageKey?: string;
  originalFilename: string;
  analysisData: AnalysisResult;
  notes: Record<string, string>;
  /**
   * Per-cluster grade assignments (e.g. "V4", "V6").
   * Keyed by `cluster_id` (stored as string keys for JSON/localStorage).
   */
  clusterGrades?: Record<string, string>;
  thumbnail?: string; // Base64 thumbnail for list view
}

export interface RouteMetadata {
  name: string;
  description?: string;
  tags: string[];
}

export interface RouteFilters {
  search?: string;
  tags?: string[];
  dateFrom?: number;
  dateTo?: number;
  sortBy?: 'newest' | 'oldest' | 'name';
}
