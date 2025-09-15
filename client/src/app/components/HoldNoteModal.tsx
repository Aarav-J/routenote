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
    <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4">
      <div
        ref={modalRef}
        className="bg-white dark:bg-gray-800 rounded-lg shadow-xl max-w-md w-full p-6"
      >
        <h3 className="text-xl font-semibold text-gray-900 dark:text-gray-100 mb-4">
          {title}
        </h3>
        
        <textarea
          ref={textareaRef}
          defaultValue={note}
          placeholder="Enter notes about this hold (texture, difficulty, tips, etc.)"
          className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md 
                   focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 
                   dark:text-gray-100 placeholder-gray-400 dark:placeholder-gray-500"
          rows={5}
        />
        
        <div className="mt-6 flex justify-end gap-3">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200 
                      rounded-lg hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 
                      transition-colors"
          >
            Save Note
          </button>
        </div>
      </div>
    </div>
  );
}