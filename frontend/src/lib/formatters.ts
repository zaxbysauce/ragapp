/**
 * Utility functions for formatting common data types
 */

/**
 * Format a file size in bytes to a human-readable string
 * @param bytes - The file size in bytes
 * @returns A formatted string like "1.5 MB" or "0 B" if bytes is falsy
 */
export function formatFileSize(bytes?: number): string {
  if (!bytes) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  let size = bytes;
  let unitIndex = 0;
  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024;
    unitIndex++;
  }
  return `${size.toFixed(1)} ${units[unitIndex]}`;
}

/**
 * Format a date string to a localized date string
 * @param dateStr - The ISO date string to format
 * @returns A localized date string or "Unknown" if dateStr is falsy
 */
export function formatDate(dateStr?: string): string {
  if (!dateStr) return "Unknown";
  return new Date(dateStr).toLocaleDateString();
}
