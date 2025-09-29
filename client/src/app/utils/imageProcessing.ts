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
  id?: string; // Unique identifier for each hold
  note?: string; // User notes for this hold
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
 * Checks if a point (x,y) is inside a bounding box
 */
export const isPointInBox = (x: number, y: number, box: number[]): boolean => {
  const [x1, y1, x2, y2] = box;
  return x >= x1 && x <= x2 && y >= y1 && y <= y2;
};

/**
 * Generate a unique ID for a hold if it doesn't have one
 */
export const ensureHoldId = (hold: BoundingBox): BoundingBox => {
  if (!hold.id) {
    const bbox = hold.bbox.join('-');
    hold.id = `hold-${hold.cluster}-${bbox}`;
  }
  return hold;
};

/**
 * Draws bounding boxes on canvas
 */
export const drawBoundingBoxes = (
  canvas: HTMLCanvasElement | null,
  image: HTMLImageElement | null,
  clusterData: AnalysisResult | null,
  selectedCluster: number | null,
  selectedHold: string | null = null,
  highlightHold: boolean = false
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
    const defaultFillColor = `rgba(${r}, ${g}, ${b}, 0.3)`;
    const selectedFillColor = `rgba(${r}, ${g}, ${b}, 0.6)`;
    const hasNoteIndicator = `rgba(255, 215, 0, 0.4)`; // Golden yellow for holds with notes
    
    // Draw each bounding box in the cluster
    cluster.items.forEach(item => {
      // Ensure each hold has an ID
      ensureHoldId(item);
      
      const [x1, y1, x2, y2] = item.bbox;
      const isSelected = selectedHold === item.id;
      const hasNote = item.note && item.note.trim().length > 0;
      
      // Choose appropriate fill color
      let fillColor = defaultFillColor;
      if (isSelected && highlightHold) {
        fillColor = selectedFillColor;
      } else if (hasNote) {
        fillColor = hasNoteIndicator; // Highlight holds that have notes
      }
      
      // Draw filled rectangle with appropriate opacity
      ctx.fillStyle = fillColor;
      ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
      
      // Draw border with full opacity (thicker for selected hold)
      ctx.strokeStyle = isSelected ? 'white' : strokeColor;
      ctx.lineWidth = isSelected ? 3 : 2;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      
      // Draw label
      ctx.fillStyle = 'white';
      ctx.strokeStyle = 'black';
      ctx.lineWidth = 2;
      ctx.font = isSelected ? 'bold 16px Arial' : '14px Arial';
      
      // Show cluster ID with note indicator if applicable
      let label = `${cluster.cluster_id}`;
      if (hasNote) {
        label += ' 📝';
      }
      
      // Draw text with outline for better visibility
      ctx.strokeText(label, x1 + 5, y1 + 16);
      ctx.fillText(label, x1 + 5, y1 + 16);
      
      // If selected and has notes, show a snippet of the note
      if (isSelected && hasNote && highlightHold) {
        const noteSnippet = item.note!.length > 20 ? item.note!.substring(0, 17) + '...' : item.note;
        
        // Draw note background
        ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        const padding = 4;
        const noteWidth = ctx.measureText(noteSnippet!).width + padding * 2;
        ctx.fillRect(x1, y1 - 24, noteWidth, 20);
        
        // Draw note text
        ctx.fillStyle = 'white';
        ctx.font = '12px Arial';
        ctx.fillText(noteSnippet!, x1 + padding, y1 - 10);
      }
    });
  });
};
