"use client";

/* eslint-disable @next/next/no-img-element */

import { useEffect, useMemo, useRef, useState } from "react";
import {
  AnalysisResult,
  BoundingBox,
  Cluster,
  drawBoundingBoxes,
  ensureHoldId,
  isPointInBox,
} from "../utils/imageProcessing";
import HoldNoteModal from "./HoldNoteModal";

type TabKey = "clusters" | "notes";
type ModeKey = "visualizer" | "grades" | "texture" | "coordinates" | "export";

const GRADE_OPTIONS = ["", "V0", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10"] as const;

function holdIdToClusterId(analysisData: AnalysisResult, holdId: string): number | null {
  for (const cluster of analysisData.clusters) {
    for (const item of cluster.items) {
      if (ensureHoldId(item).id === holdId) return cluster.cluster_id;
    }
  }
  return null;
}

export default function WorkspaceViewer(props: {
  projectName: string;
  originalImage: string;
  analysisData: AnalysisResult;
  notes: Record<string, string>;
  onNotesChange: (notes: Record<string, string>) => void;
  clusterGrades: Record<string, string>;
  onClusterGradesChange: (grades: Record<string, string>) => void;
  onSaveRoute?: () => void;
}) {
  const { projectName, originalImage, analysisData, notes, onNotesChange, clusterGrades, onClusterGradesChange, onSaveRoute } = props;

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const [imageLoaded, setImageLoaded] = useState(false);

  const [mode, setMode] = useState<ModeKey>("visualizer");
  const [tab, setTab] = useState<TabKey>("clusters");
  const [selectedCluster, setSelectedCluster] = useState<number | null>(null);
  const [selectedHold, setSelectedHold] = useState<string | null>(null);
  const [showNoteModal, setShowNoteModal] = useState(false);

  const totalHolds = useMemo(
    () => analysisData.clusters.reduce((sum, c) => sum + (c.count ?? c.items?.length ?? 0), 0),
    [analysisData.clusters]
  );

  const holdNotes = useMemo(() => notes || {}, [notes]);
  const gradesMap = useMemo(() => clusterGrades || {}, [clusterGrades]);

  // Apply notes to analysisData in-memory (for drawBoundingBoxes note indicators).
  useEffect(() => {
    analysisData.clusters.forEach((cluster) => {
      cluster.items.forEach((item) => {
        const id = ensureHoldId(item).id as string;
        const n = holdNotes[id];
        item.note = n;
      });
    });
  }, [analysisData, holdNotes]);

  useEffect(() => {
    if (!imageLoaded) return;
    drawBoundingBoxes(canvasRef.current, imageRef.current, analysisData, selectedCluster, selectedHold, true);
  }, [analysisData, selectedCluster, selectedHold, imageLoaded, holdNotes]);

  const currentNote = useMemo(() => {
    if (!selectedHold) return "";
    return holdNotes[selectedHold] || "";
  }, [holdNotes, selectedHold]);

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!analysisData || !imageRef.current || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;

    let found: BoundingBox | null = null;
    let foundClusterId: number | null = null;
    for (const cluster of analysisData.clusters) {
      if (selectedCluster !== null && cluster.cluster_id !== selectedCluster) continue;
      for (const item of cluster.items) {
        if (isPointInBox(x, y, item.bbox)) {
          found = item;
          foundClusterId = cluster.cluster_id;
        }
      }
    }

    if (!found) {
      setSelectedHold(null);
      return;
    }

    const id = ensureHoldId(found).id as string;
    if (selectedHold === id) {
      if (mode === "visualizer") setShowNoteModal(true);
      return;
    }
    setSelectedHold(id);
    if (mode === "visualizer") setTab("notes");
    if (mode === "grades" && foundClusterId !== null) {
      setSelectedCluster(foundClusterId);
    }
  };

  const saveNote = (noteText: string) => {
    if (!selectedHold) return;
    const updated = { ...holdNotes };
    if (noteText.trim() === "") {
      delete updated[selectedHold];
    } else {
      updated[selectedHold] = noteText;
    }
    onNotesChange(updated);
  };

  const clusters: Cluster[] = analysisData.clusters || [];

  const notesList = useMemo(() => {
    return Object.entries(holdNotes)
      .map(([holdId, note]) => ({
        holdId,
        note,
        clusterId: holdIdToClusterId(analysisData, holdId),
      }))
      .sort((a, b) => (a.clusterId ?? 9999) - (b.clusterId ?? 9999));
  }, [analysisData, holdNotes]);

  return (
    <div className="flex h-[calc(100vh-4rem)] overflow-hidden bg-[#0e0e0f] text-white">
      <aside className="fixed bottom-0 left-0 top-16 flex w-64 flex-col gap-2 bg-[#131314] p-4 text-sm font-medium">
        <div className="mb-6 px-2">
          <div className="flex items-center gap-3 rounded-xl bg-[#1a191b] p-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-[#cc97ff]/10">
              <span className="text-[#cc97ff]">◎</span>
            </div>
            <div>
              <p className="text-xs font-bold uppercase tracking-wider text-[#cc97ff]">Intelligence</p>
              <p className="text-[10px] text-[#adaaab]">AI Route Analysis</p>
            </div>
          </div>
        </div>

        <nav className="flex flex-1 flex-col gap-1">
          <button
            type="button"
            onClick={() => setMode("visualizer")}
            className={`flex items-center gap-3 rounded-lg px-4 py-3 transition-all ${
              mode === "visualizer"
                ? "translate-x-1 bg-[#1a191b] text-[#cc97ff] shadow-[inset_0_0_10px_rgba(204,151,255,0.1)]"
                : "text-[#adaaab] hover:bg-[#1a191b] hover:text-white"
            }`}
          >
            <span>◎</span>
            <span>Visualizer</span>
          </button>
          <button
            type="button"
            onClick={() => setMode("grades")}
            className={`flex items-center gap-3 rounded-lg px-4 py-3 transition-all ${
              mode === "grades"
                ? "translate-x-1 bg-[#1a191b] text-[#cc97ff] shadow-[inset_0_0_10px_rgba(204,151,255,0.1)]"
                : "text-[#adaaab] hover:bg-[#1a191b] hover:text-white"
            }`}
          >
            <span className="h-2 w-2 rounded-full bg-[#262627]" />
            <span>Grades</span>
          </button>
          {[
            { key: "texture" as const, label: "Texture" },
            { key: "coordinates" as const, label: "Coordinates" },
            { key: "export" as const, label: "Export" },
          ].map((item) => (
            <button
              key={item.key}
              type="button"
              onClick={() => setMode(item.key)}
              className={`flex items-center gap-3 rounded-lg px-4 py-3 transition-all ${
                mode === item.key
                  ? "translate-x-1 bg-[#1a191b] text-[#cc97ff] shadow-[inset_0_0_10px_rgba(204,151,255,0.1)]"
                  : "text-[#adaaab] hover:bg-[#1a191b] hover:text-white"
              }`}
            >
              <span className="h-2 w-2 rounded-full bg-[#262627]" />
              <span>{item.label}</span>
            </button>
          ))}
        </nav>

        <div className="mt-auto p-2">
          <button
            type="button"
            onClick={onSaveRoute}
            className="flex w-full items-center justify-center gap-2 rounded-xl bg-[#cc97ff] px-4 py-3 font-bold text-black transition-all hover:shadow-[0_0_20px_rgba(204,151,255,0.3)] active:scale-95"
          >
            <span className="text-sm">⚡</span>
            Save Route
          </button>
        </div>
      </aside>

      <main className="ml-64 flex flex-1 overflow-hidden">
        <section className="relative flex flex-1 flex-col p-6">
          <div className="mb-4 flex items-center justify-between">
            <div className="flex flex-col">
              <h1 className="font-headline text-2xl font-bold tracking-tight text-white">
                Project: {projectName}
              </h1>
              <p className="text-sm text-[#adaaab]">Scan • Workspace analysis</p>
            </div>
            <div className="flex items-center gap-2 rounded-full border border-[#484849]/30 bg-[#1a191b] px-3 py-1">
              <span className="h-2 w-2 rounded-full bg-[#69f6b8] shadow-[0_0_8px_#69f6b8]" />
              <span className="text-xs font-medium text-[#69f6b8]">AI Live View</span>
            </div>
          </div>

          <div className="group relative flex-1 overflow-hidden rounded-xl border border-[#484849]/20 bg-[#131314]">
            <div className="absolute inset-0 flex items-center justify-center p-4">
              <div className="relative inline-block max-h-full max-w-full">
                <img
                  ref={imageRef}
                  src={originalImage}
                  alt="Climbing wall analysis"
                  className="block max-h-[calc(100vh-8rem)] max-w-full rounded-lg opacity-90 transition-opacity duration-700 group-hover:opacity-100"
                  onLoad={() => setImageLoaded(true)}
                />
                <canvas
                  ref={canvasRef}
                  className="absolute left-0 top-0 cursor-crosshair rounded-lg"
                  onClick={handleCanvasClick}
                />
              </div>
            </div>

            <div className="pointer-events-none absolute left-4 top-4 flex flex-col gap-1 font-mono text-[10px] text-[#adaaab]">
              <span className="rounded bg-black/40 px-2 py-0.5">Holds: {totalHolds}</span>
              {selectedHold && (
                <span className="rounded bg-black/40 px-2 py-0.5">Selected: {selectedHold}</span>
              )}
            </div>
          </div>
        </section>

        <aside className="flex w-[380px] flex-col overflow-hidden border-l border-[#484849]/20 bg-[#131314]">
          {mode === "visualizer" && (
            <>
              <div className="flex border-b border-[#484849]/20">
                <button
                  type="button"
                  onClick={() => setTab("clusters")}
                  className={`flex-1 py-4 text-sm ${
                    tab === "clusters"
                      ? "border-b-2 border-[#cc97ff] bg-[#cc97ff]/5 font-bold text-[#cc97ff]"
                      : "font-medium text-[#adaaab] hover:bg-[#1a191b] hover:text-white"
                  }`}
                >
                  Color Clusters
                </button>
                <button
                  type="button"
                  onClick={() => setTab("notes")}
                  className={`flex-1 py-4 text-sm ${
                    tab === "notes"
                      ? "border-b-2 border-[#cc97ff] bg-[#cc97ff]/5 font-bold text-[#cc97ff]"
                      : "font-medium text-[#adaaab] hover:bg-[#1a191b] hover:text-white"
                  }`}
                >
                  Hold Notes
                </button>
              </div>

              {tab === "clusters" ? (
                <div className="flex-1 space-y-6 overflow-y-auto p-5">
              <div className="flex items-center justify-between">
                <h3 className="text-xs font-bold uppercase tracking-widest text-[#adaaab]">Detection Summary</h3>
                <span className="rounded-full bg-[#cc97ff]/10 px-2 py-0.5 text-[10px] text-[#cc97ff]">
                  {totalHolds} Holds Found
                </span>
              </div>

              <div className="grid grid-cols-1 gap-3">
                {clusters.map((cluster) => {
                  const [r, g, b] = cluster.rgb_color;
                  const color = `rgb(${r}, ${g}, ${b})`;
                  const isSelected = selectedCluster === cluster.cluster_id;
                  const pct = totalHolds > 0 ? Math.round(((cluster.count || cluster.items.length) / totalHolds) * 100) : 0;
                  return (
                    <button
                      key={cluster.cluster_id}
                      type="button"
                      onClick={() => setSelectedCluster((prev) => (prev === cluster.cluster_id ? null : cluster.cluster_id))}
                      className={`group cursor-pointer rounded-xl border bg-[#1a191b] p-4 text-left transition-all ${
                        isSelected ? "border-[#cc97ff]/60" : "border-[#484849]/20 hover:border-[#cc97ff]/40"
                      }`}
                    >
                      <div className="mb-3 flex items-center justify-between">
                        <div className="flex items-center gap-3">
                          <div
                            className="h-8 w-8 rounded-full"
                            style={{ backgroundColor: color, boxShadow: `0 0 12px rgba(${r},${g},${b},0.25)` }}
                          />
                          <div>
                            <p className="text-sm font-bold text-white">Cluster {cluster.cluster_id}</p>
                            <p className="text-[10px] text-[#adaaab]">Color group</p>
                          </div>
                        </div>
                        <span className="font-headline text-lg font-bold" style={{ color }}>
                          {cluster.count || cluster.items.length}
                        </span>
                      </div>
                      <div className="h-1 w-full overflow-hidden rounded-full bg-[#262627]">
                        <div className="h-full" style={{ width: `${pct}%`, backgroundColor: color }} />
                      </div>
                      <div className="mt-2 text-[10px] text-[#767576]">{pct}% of holds</div>
                    </button>
                  );
                })}
              </div>
                </div>
              ) : (
                <div className="flex-1 space-y-4 overflow-y-auto p-5">
              <div className="flex items-center justify-between">
                <h3 className="text-xs font-bold uppercase tracking-widest text-[#adaaab]">Hold Notes</h3>
                <span className="rounded-full bg-[#262627] px-2 py-0.5 text-[10px] text-[#adaaab]">
                  {Object.keys(holdNotes).length} noted
                </span>
              </div>

              {selectedHold ? (
                <div className="rounded-2xl border border-[#484849]/30 bg-[#1a191b] p-4">
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <p className="text-[10px] font-bold uppercase tracking-widest text-[#adaaab]">Selected Hold</p>
                      <p className="mt-1 text-sm font-semibold text-white">{selectedHold}</p>
                      <p className="mt-1 text-xs text-[#adaaab]">
                        Cluster {holdIdToClusterId(analysisData, selectedHold) ?? "?"}
                      </p>
                    </div>
                    <button
                      type="button"
                      onClick={() => setShowNoteModal(true)}
                      className="rounded-lg bg-gradient-to-r from-[#cc97ff] to-[#9c48ea] px-3 py-2 text-xs font-bold text-black"
                    >
                      {currentNote ? "Edit Note" : "Add Note"}
                    </button>
                  </div>

                  {currentNote ? (
                    <div className="mt-3 rounded-xl border border-[#484849]/20 bg-black/20 p-3 text-sm text-white/90">
                      <p className="whitespace-pre-wrap leading-relaxed">{currentNote}</p>
                    </div>
                  ) : (
                    <p className="mt-3 text-sm text-[#adaaab]">No note for this hold yet. Click “Add Note”.</p>
                  )}
                </div>
              ) : (
                <div className="rounded-2xl border border-[#484849]/20 bg-[#1a191b] p-4 text-sm text-[#adaaab]">
                  Click a hold in the canvas to open its note editor here.
                </div>
              )}

              {notesList.length > 0 && (
                <div className="space-y-3">
                  {notesList.map((n) => (
                    <button
                      key={n.holdId}
                      type="button"
                      onClick={() => {
                        setSelectedHold(n.holdId);
                        const clusterId = n.clusterId;
                        if (clusterId != null) setSelectedCluster(clusterId);
                        setShowNoteModal(true);
                      }}
                      className="w-full rounded-xl border border-[#484849]/20 bg-[#1a191b] p-4 text-left transition hover:border-[#cc97ff]/30"
                    >
                      <div className="flex items-center justify-between gap-3">
                        <div>
                          <p className="text-[10px] font-bold uppercase tracking-widest text-[#adaaab]">
                            Cluster {n.clusterId ?? "?"}
                          </p>
                          <p className="mt-1 text-sm font-semibold text-white">{n.holdId}</p>
                        </div>
                        <span className="text-xs text-[#cc97ff]">Edit</span>
                      </div>
                      <p className="mt-2 line-clamp-3 text-sm text-[#adaaab]">{n.note}</p>
                    </button>
                  ))}
                </div>
              )}
                </div>
              )}
            </>
          )}

          {mode === "grades" && (
            <div className="flex-1 space-y-4 overflow-y-auto p-5">
              <div className="flex items-center justify-between">
                <h3 className="text-xs font-bold uppercase tracking-widest text-[#adaaab]">Grades</h3>
                <span className="rounded-full bg-[#262627] px-2 py-0.5 text-[10px] text-[#adaaab]">
                  {Object.keys(gradesMap).filter((k) => gradesMap[k]).length} graded
                </span>
              </div>

              <div className="rounded-2xl border border-[#484849]/20 bg-[#1a191b] p-4 text-sm text-[#adaaab]">
                Click a hold in the canvas (or click a cluster) then assign a V-grade to that cluster. Grades are saved to this route.
              </div>

              {selectedCluster !== null ? (
                <div className="rounded-2xl border border-[#484849]/30 bg-[#1a191b] p-4">
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <p className="text-[10px] font-bold uppercase tracking-widest text-[#adaaab]">Selected Cluster</p>
                      <p className="mt-1 text-sm font-semibold text-white">Cluster {selectedCluster}</p>
                      <p className="mt-1 text-xs text-[#adaaab]">{clusters.find((c) => c.cluster_id === selectedCluster)?.count ?? 0} holds</p>
                    </div>
                  </div>

                  <div className="mt-4 space-y-2">
                    <label className="ml-1 text-xs font-bold uppercase tracking-widest text-[#adaaab]">V Grade</label>
                    <select
                      value={gradesMap[String(selectedCluster)] || ""}
                      onChange={(e) => {
                        const value = e.target.value;
                        const updated = { ...gradesMap };
                        if (!value) {
                          delete updated[String(selectedCluster)];
                        } else {
                          updated[String(selectedCluster)] = value;
                        }
                        onClusterGradesChange(updated);
                      }}
                      className="h-12 w-full rounded-xl border border-[#484849]/30 bg-[#201f21] px-4 text-sm text-white focus:outline-none focus:ring-2 focus:ring-[#cc97ff]/40"
                    >
                      <option value="">Unrated</option>
                      {GRADE_OPTIONS.filter(Boolean).map((g) => (
                        <option key={g} value={g}>
                          {g}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
              ) : (
                <div className="rounded-2xl border border-[#484849]/20 bg-[#1a191b] p-4 text-sm text-[#adaaab]">
                  No cluster selected. Click a hold (or choose a cluster below) to grade it.
                </div>
              )}

              <div className="rounded-2xl border border-[#484849]/20 bg-[#1a191b] p-4">
                <p className="text-[10px] font-bold uppercase tracking-widest text-[#adaaab]">Clusters</p>
                <div className="mt-3 space-y-2">
                  {clusters.map((c) => {
                    const grade = gradesMap[String(c.cluster_id)] || "";
                    const isSelected = selectedCluster === c.cluster_id;
                    return (
                      <button
                        key={c.cluster_id}
                        type="button"
                        onClick={() => setSelectedCluster(c.cluster_id)}
                        className={`flex w-full items-center justify-between rounded-xl border px-4 py-3 text-left transition ${
                          isSelected ? "border-[#cc97ff]/60 bg-black/20" : "border-[#484849]/20 hover:border-[#cc97ff]/30 hover:bg-black/10"
                        }`}
                      >
                        <div className="flex items-center gap-3">
                          <div
                            className="h-3 w-3 rounded-full"
                            style={{ backgroundColor: `rgb(${c.rgb_color[0]},${c.rgb_color[1]},${c.rgb_color[2]})` }}
                          />
                          <div>
                            <div className="text-sm font-semibold text-white">Cluster {c.cluster_id}</div>
                            <div className="text-[10px] text-[#adaaab]">{c.count} holds</div>
                          </div>
                        </div>
                        <div className="rounded-full bg-[#262627] px-2 py-1 text-[10px] font-bold text-white">
                          {grade || "Unrated"}
                        </div>
                      </button>
                    );
                  })}
                </div>
              </div>

              <div className="rounded-2xl border border-[#484849]/20 bg-[#1a191b] p-4">
                <p className="text-[10px] font-bold uppercase tracking-widest text-[#adaaab]">Summary</p>
                <div className="mt-3 grid grid-cols-2 gap-3">
                  {["V0", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10"].map((g) => {
                    const count = Object.values(gradesMap).filter((v) => v === g).length;
                    return (
                      <div key={g} className="rounded-xl border border-[#484849]/20 bg-black/20 p-3">
                        <div className="text-xs font-semibold text-white">{g}</div>
                        <div className="mt-1 text-sm text-[#adaaab]">{count} clusters</div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          )}

          {mode !== "visualizer" && mode !== "grades" && (
            <div className="flex flex-1 items-center justify-center p-8 text-sm text-[#adaaab]">
              {mode === "texture" && "Texture tools coming next."}
              {mode === "coordinates" && "Coordinate tools coming next."}
              {mode === "export" && "Export tools coming next."}
            </div>
          )}
        </aside>
      </main>

      <HoldNoteModal
        isOpen={showNoteModal}
        onClose={() => setShowNoteModal(false)}
        note={selectedHold ? holdNotes[selectedHold] || "" : ""}
        onSave={saveNote}
        title={
          selectedHold
            ? `Add Note for Hold ${selectedHold} (Cluster ${holdIdToClusterId(analysisData, selectedHold) ?? "?"})`
            : "Add Note for Hold"
        }
      />
    </div>
  );
}

