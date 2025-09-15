/**
 * Utility functions for processing and visualizing climbing wall images
 */

// Interface for bounding box data
export interface BoundingBox {
  bbox: number[];
  conf: number;
  class_id: number;
  class_name: string;
  color_name?: string;
  hsv?: number[];
  cluster: number;
}

// Interface for cluster data
export interface Cluster {
  cluster_id: number;
  count: number;
  avg_hsv: number[];
  rgb_color: number[];
  items: BoundingBox[];
}

// Interface for the full analysis results
export interface AnalysisResult {
  clusters: Cluster[];
  processedImageUrl?: string;
  legendImageUrl?: string;
  filename?: string;
}

/**
 * Determines if a color is dark (for choosing appropriate text color)
 */
export const isDarkColor = (rgb: number[]): boolean => {
  const [r, g, b] = rgb;
  // Calculate relative luminance - colors with luminance < 0.5 are considered dark
  const luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
  return luminance < 0.5;
};

/**
 * Draws bounding boxes on canvas
 */
export const drawBoundingBoxes = (
  canvas: HTMLCanvasElement | null,
  image: HTMLImageElement | null,
  clusterData: AnalysisResult | null,
  selectedCluster: number | null
): void => {
  if (!clusterData || !image || !canvas) return;
  
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  
  // Important: Set canvas dimensions to match the original image dimensions
  // This ensures coordinates match exactly with the detection results
  canvas.width = image.naturalWidth;
  canvas.height = image.naturalHeight;
  
  // Make the canvas size match the displayed image size with CSS
  canvas.style.width = `${image.width}px`;
  canvas.style.height = `${image.height}px`;
  
  // Clear previous drawings
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  
  // Draw bounding boxes for each cluster
  clusterData.clusters.forEach(cluster => {
    // Skip if not the selected cluster (when one is selected)
    if (selectedCluster !== null && selectedCluster !== cluster.cluster_id) {
      return;
    }
    
    // Get RGB color for this cluster
    const [r, g, b] = cluster.rgb_color;
    const strokeColor = `rgb(${r}, ${g}, ${b})`;
    const fillColor = `rgba(${r}, ${g}, ${b}, 0.3)`;
    
    // Draw each bounding box in the cluster
    cluster.items.forEach(item => {
      const [x1, y1, x2, y2] = item.bbox;
      
      // Draw filled rectangle with low opacity
      ctx.fillStyle = fillColor;
      ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
      
      // Draw border with full opacity
      ctx.strokeStyle = strokeColor;
      ctx.lineWidth = 2;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      
      // Draw label
      ctx.fillStyle = 'white';
      ctx.strokeStyle = 'black';
      ctx.lineWidth = 3;
      ctx.font = '14px Arial';
      const label = `${cluster.cluster_id}`;
      
      // Draw text with outline for better visibility
      ctx.strokeText(label, x1, y1 - 5);
      ctx.fillText(label, x1, y1 - 5);
    });
  });
};