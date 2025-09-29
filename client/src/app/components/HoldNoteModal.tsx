"use client";

import { useEffect, useRef } from "react";

interface HoldNoteModalProps {
  isOpen: boolean;
  onClose: () => void;
  note: string;
  onSave: (note: string) => void;
  title?: string;
}

export default function HoldNoteModal({
  isOpen,
  onClose,
  note,
  onSave,
  title = "Add Note for Hold"
}: HoldNoteModalProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const modalRef = useRef<HTMLDivElement>(null);
  
  // Focus the textarea when modal opens
  useEffect(() => {
    if (isOpen && textareaRef.current) {
      textareaRef.current.focus();
    }
  }, [isOpen]);
  
  // Handle click outside to close
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (modalRef.current && !modalRef.current.contains(event.target as Node)) {
        onClose();
      }
    }
    
    if (isOpen) {
      document.addEventListener("mousedown", handleClickOutside);
    }
    
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [isOpen, onClose]);
  
  // Handle Escape key to close
  useEffect(() => {
    function handleEscKey(event: KeyboardEvent) {
      if (event.key === "Escape") {
        onClose();
      }
    }
    
    if (isOpen) {
      document.addEventListener("keydown", handleEscKey);
    }
    
    return () => {
      document.removeEventListener("keydown", handleEscKey);
    };
  }, [isOpen, onClose]);
  
  // Handle save action
  const handleSave = () => {
    if (textareaRef.current) {
      onSave(textareaRef.current.value);
    }
    onClose();
  };
  
  if (!isOpen) return null;
  
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 px-4 py-10 backdrop-blur-sm">
      <div
        ref={modalRef}
        className="w-full max-w-md rounded-3xl border border-[var(--border)] bg-[color-mix(in_srgb,var(--background-raised)_92%,_black_8%)] p-7 text-white shadow-[0_20px_60px_rgba(5,8,14,0.65)]"
      >
        <h3 className="text-xl font-semibold text-white">
          {title}
        </h3>
        
        <textarea
          ref={textareaRef}
          defaultValue={note}
          placeholder="Enter notes about this hold (texture, difficulty, tips, etc.)"
          className="mt-4 w-full rounded-2xl border border-white/10 bg-black/25 px-4 py-3 text-sm text-white placeholder:text-[var(--foreground-muted)] focus:border-[var(--primary)] focus:outline-none focus:ring-2 focus:ring-[var(--primary)]/60"
          rows={5}
        />
        
        <div className="mt-6 flex justify-end gap-3">
          <button
            onClick={onClose}
            className="inline-flex items-center gap-2 rounded-full border border-white/15 px-5 py-2 text-sm font-semibold text-white transition hover:border-[var(--primary)]/50 hover:text-[var(--primary)]"
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            className="inline-flex items-center gap-2 rounded-full bg-[var(--primary)] px-5 py-2 text-sm font-semibold text-white shadow-[0_12px_28px_rgba(197,24,241,0.35)] transition hover:bg-[var(--primary-strong)]"
          >
            Save Note
          </button>
        </div>
      </div>
    </div>
  );
}
