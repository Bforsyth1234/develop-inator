import { render, screen, waitFor } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import UserProfile from "../UserProfile";

describe("UserProfile", () => {
  it("shows loading state initially", () => {
    render(<UserProfile />);
    expect(screen.getByText("Loading profile...")).toBeInTheDocument();
  });

  it("renders profile data after loading", async () => {
    render(<UserProfile />);
    await waitFor(() => {
      expect(screen.getByTestId("profile-name")).toHaveTextContent("Jane Doe");
    });
    expect(screen.getByTestId("profile-email")).toHaveTextContent(
      "jane@example.com"
    );
    expect(screen.getByTestId("profile-role")).toHaveTextContent("Developer");
  });
});

