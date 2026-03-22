import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import Header from "../Header";

describe("Header", () => {
  it("renders the app title", () => {
    render(<Header />);
    expect(screen.getByText("Dummy React App")).toBeInTheDocument();
  });
});

