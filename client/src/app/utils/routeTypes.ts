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
  originalFilename: string;
  analysisData: AnalysisResult;
  notes: Record<string, string>;
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
