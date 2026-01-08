"use client";

import { useState, useEffect, useRef } from "react";
import { RouteMetadata } from "../utils/routeTypes";

interface RouteFormProps {
  isOpen: boolean;
  onClose: () => void;
  onSubmit: (metadata: RouteMetadata) => void;
  initialData?: RouteMetadata;
  defaultName?: string;
}

export default function RouteForm({
  isOpen,
  onClose,
  onSubmit,
  initialData,
  defaultName
}: RouteFormProps) {
  const [name, setName] = useState(initialData?.name || defaultName || '');
  const [description, setDescription] = useState(initialData?.description || '');
  const [tags, setTags] = useState<string[]>(initialData?.tags || []);
  const [tagInput, setTagInput] = useState('');
  const formRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (isOpen) {
      setName(initialData?.name || defaultName || '');
      setDescription(initialData?.description || '');
      setTags(initialData?.tags || []);
      setTagInput('');
    }
  }, [isOpen, initialData, defaultName]);

  // Handle click outside to close
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (formRef.current && !formRef.current.contains(event.target as Node)) {
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

  // Handle Escape key
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

  const handleAddTag = () => {
    const trimmed = tagInput.trim();
    if (trimmed && !tags.includes(trimmed)) {
      setTags([...tags, trimmed]);
      setTagInput('');
    }
  };

  const handleRemoveTag = (tagToRemove: string) => {
    setTags(tags.filter(tag => tag !== tagToRemove));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit({
      name: name.trim() || `Route ${new Date().toLocaleDateString()}`,
      description: description.trim() || undefined,
      tags
    });
    onClose();
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 px-4 py-10 backdrop-blur-sm">
      <div
        ref={formRef}
        className="w-full max-w-lg rounded-2xl border border-[var(--border)] bg-[var(--card-background)] p-6 shadow-[var(--shadow-primary-strong)]"
      >
        <div className="mb-6 flex items-center justify-between">
          <h2 className="text-xl font-bold text-white">Save Route</h2>
          <button
            onClick={onClose}
            className="rounded-lg p-2 text-[var(--foreground-muted)] transition hover:bg-[var(--background-raised-soft)] hover:text-white"
            aria-label="Close"
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <form onSubmit={handleSubmit} className="space-y-5">
          {/* Route Name */}
          <div>
            <label htmlFor="route-name" className="mb-2 block text-sm font-medium text-white">
              Route Name <span className="text-[var(--foreground-muted)]">*</span>
            </label>
            <input
              id="route-name"
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder={`Route ${new Date().toLocaleDateString()}`}
              className="w-full rounded-lg border border-[var(--border)] bg-[var(--background-raised)] px-4 py-2.5 text-white placeholder:text-[var(--foreground-subtle)] focus:border-[var(--primary)] focus:outline-none focus:ring-2 focus:ring-[var(--primary-soft)]"
              autoFocus
              required
            />
          </div>

          {/* Description */}
          <div>
            <label htmlFor="route-description" className="mb-2 block text-sm font-medium text-white">
              Description
            </label>
            <textarea
              id="route-description"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Add notes about this route..."
              rows={3}
              className="w-full rounded-lg border border-[var(--border)] bg-[var(--background-raised)] px-4 py-2.5 text-sm text-white placeholder:text-[var(--foreground-subtle)] focus:border-[var(--primary)] focus:outline-none focus:ring-2 focus:ring-[var(--primary-soft)] resize-none"
            />
          </div>

          {/* Tags */}
          <div>
            <label htmlFor="route-tags" className="mb-2 block text-sm font-medium text-white">
              Tags
            </label>
            <div className="flex gap-2">
              <input
                id="route-tags"
                type="text"
                value={tagInput}
                onChange={(e) => setTagInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    e.preventDefault();
                    handleAddTag();
                  }
                }}
                placeholder="Add a tag and press Enter"
                className="flex-1 rounded-lg border border-[var(--border)] bg-[var(--background-raised)] px-4 py-2.5 text-sm text-white placeholder:text-[var(--foreground-subtle)] focus:border-[var(--primary)] focus:outline-none focus:ring-2 focus:ring-[var(--primary-soft)]"
              />
              <button
                type="button"
                onClick={handleAddTag}
                className="rounded-lg border border-[var(--border)] bg-[var(--background-raised)] px-4 py-2.5 text-sm font-medium text-white transition hover:border-[var(--primary)] hover:bg-[var(--primary-soft)]"
              >
                Add
              </button>
            </div>
            {tags.length > 0 && (
              <div className="mt-3 flex flex-wrap gap-2">
                {tags.map((tag) => (
                  <span
                    key={tag}
                    className="group inline-flex items-center gap-1.5 rounded-full border border-[var(--primary-soft)] bg-[var(--primary-softer)] px-3 py-1 text-xs font-medium text-[var(--primary-light)]"
                  >
                    {tag}
                    <button
                      type="button"
                      onClick={() => handleRemoveTag(tag)}
                      className="text-[var(--primary-light)] opacity-60 transition hover:opacity-100"
                    >
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                      </svg>
                    </button>
                  </span>
                ))}
              </div>
            )}
          </div>

          {/* Form Actions */}
          <div className="flex justify-end gap-3 pt-2">
            <button
              type="button"
              onClick={onClose}
              className="rounded-lg border border-[var(--border)] bg-[var(--background-raised)] px-5 py-2.5 text-sm font-semibold text-white transition hover:border-[var(--primary)] hover:text-[var(--primary-light)]"
            >
              Cancel
            </button>
            <button
              type="submit"
              className="rounded-lg bg-gradient-to-r from-[var(--primary)] to-[var(--primary-light)] px-5 py-2.5 text-sm font-semibold text-white shadow-[var(--shadow-primary)] transition hover:shadow-[var(--shadow-primary-strong)]"
            >
              Save Route
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
