import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { NavigationRail } from "./NavigationRail";
import type { NavItemId } from "./navigationTypes";

// Mock health status
const mockHealthStatus = {
  backend: true,
  embeddings: true,
  chat: true,
  loading: false,
};

// Mock window.location.href for chatNew navigation
const mockLocation = { href: "" };
Object.defineProperty(window, "location", {
  value: mockLocation,
  writable: true,
});

describe("NavigationRail", () => {
  describe("Navigation Items", () => {
    it("renders all navigation items including Modern Chat", () => {
      render(
        <NavigationRail
          activeItem="chat"
          onItemSelect={vi.fn()}
          healthStatus={mockHealthStatus}
        />
      );

      // Check for all expected nav items
      expect(screen.getByLabelText("Chat")).toBeInTheDocument();
      expect(screen.getByLabelText("Modern Chat")).toBeInTheDocument();
      expect(screen.getByLabelText("Documents")).toBeInTheDocument();
      expect(screen.getByLabelText("Memory")).toBeInTheDocument();
      expect(screen.getByLabelText("Vaults")).toBeInTheDocument();
      expect(screen.getByLabelText("Settings")).toBeInTheDocument();
    });

    it("renders Modern Chat without the legacy NEW badge", () => {
      render(
        <NavigationRail
          activeItem="chat"
          onItemSelect={vi.fn()}
          healthStatus={mockHealthStatus}
        />
      );

      expect(screen.queryByText("NEW")).not.toBeInTheDocument();
    });

    it("renders Modern Chat with gradient styling", () => {
      render(
        <NavigationRail
          activeItem="chat"
          onItemSelect={vi.fn()}
          healthStatus={mockHealthStatus}
        />
      );

      const chatNewButton = screen.getByLabelText("Modern Chat");
      expect(chatNewButton).toHaveClass("bg-gradient-to-br", "from-purple-500/10", "to-pink-500/10");
      expect(chatNewButton).toHaveClass("border", "border-purple-500/30");
    });

    it("renders Modern Chat icon with gradient background", () => {
      render(
        <NavigationRail
          activeItem="chat"
          onItemSelect={vi.fn()}
          healthStatus={mockHealthStatus}
        />
      );

      const chatNewIconContainer = screen.getByLabelText("Modern Chat").querySelector("div");
      expect(chatNewIconContainer).toHaveClass("bg-gradient-to-br", "from-purple-500", "to-pink-500");
      expect(chatNewIconContainer).toHaveClass("text-white");
    });

    it("renders Modern Chat label with gradient text", () => {
      render(
        <NavigationRail
          activeItem="chat"
          onItemSelect={vi.fn()}
          healthStatus={mockHealthStatus}
        />
      );

      // Get the visible label by querying the button and filtering out sr-only
      const chatNewButton = screen.getByLabelText("Modern Chat");
      const allLabels = chatNewButton.querySelectorAll("span");
      // The visible label is the one without sr-only class
      const visibleLabels = Array.from(allLabels).filter((span) => !span.classList.contains("sr-only"));
      const chatNewLabel = visibleLabels.find((span) => span.textContent === "Modern Chat");
      
      expect(chatNewLabel).toBeInTheDocument();
      expect(chatNewLabel).toHaveClass("bg-gradient-to-r", "from-purple-500", "to-pink-500");
      expect(chatNewLabel).toHaveClass("bg-clip-text", "text-transparent");
    });
  });

  describe("Active State", () => {
    it("highlights active item with primary background", () => {
      render(
        <NavigationRail
          activeItem="chat"
          onItemSelect={vi.fn()}
          healthStatus={mockHealthStatus}
        />
      );

      const chatButton = screen.getByLabelText("Chat");
      expect(chatButton).toHaveClass("bg-primary/10");
    });

    it("does not show active indicator on chatNew when active", () => {
      render(
        <NavigationRail
          activeItem="chatNew"
          onItemSelect={vi.fn()}
          healthStatus={mockHealthStatus}
        />
      );

      // chatNew should NOT have the right-side active indicator
      const chatNewButton = screen.getByLabelText("Modern Chat");
      const activeIndicator = chatNewButton.querySelector("span.w-1.h-4");
      expect(activeIndicator).not.toBeInTheDocument();
    });
  });

  describe("Interactions", () => {
    it("calls onItemSelect for regular items", () => {
      const handleSelect = vi.fn<(id: NavItemId) => void>();

      render(
        <NavigationRail
          activeItem="chat"
          onItemSelect={handleSelect}
          healthStatus={mockHealthStatus}
        />
      );

      const documentsButton = screen.getByLabelText("Documents");
      fireEvent.click(documentsButton);

      expect(handleSelect).toHaveBeenCalledWith("documents");
    });

    it("navigates to /chat/redesign for Modern Chat", () => {
      render(
        <NavigationRail
          activeItem="chat"
          onItemSelect={vi.fn()}
          healthStatus={mockHealthStatus}
        />
      );

      const chatNewButton = screen.getByLabelText("Modern Chat");
      fireEvent.click(chatNewButton);

      expect(window.location.href).toBe("/chat/redesign");
    });
  });

  describe("Health Status Footer", () => {
    it("renders health status indicators", () => {
      render(
        <NavigationRail
          activeItem="chat"
          onItemSelect={vi.fn()}
          healthStatus={mockHealthStatus}
        />
      );

      // Check for health status labels in the footer
      expect(screen.getByText("API")).toBeInTheDocument();
      expect(screen.getByText("Embeddings")).toBeInTheDocument();
      // The "Chat" text appears both in nav and health status, so use getAllByText
      const chatLabels = screen.getAllByText("Chat");
      expect(chatLabels.length).toBeGreaterThan(0);

      // Check green indicators (healthy) - look in the footer container by class
      const footerContainer = screen.getByText("API").closest("div");
      const greenIndicators = footerContainer.querySelectorAll(".bg-green-500");
      expect(greenIndicators.length).toBeGreaterThan(0);
    });
  });

  describe("TypeScript Types", () => {
    it("chatNew is a valid NavItemId", () => {
      // This test verifies the type is properly exported
      const validId: NavItemId = "chatNew";
      expect(validId).toBe("chatNew");
    });
  });
});
