import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import Auth from "../Auth";

describe("Auth", () => {
  it("renders the login form", () => {
    render(<Auth />);
    expect(screen.getByRole("heading", { name: "Login" })).toBeInTheDocument();
    expect(screen.getByTestId("username-input")).toBeInTheDocument();
    expect(screen.getByTestId("password-input")).toBeInTheDocument();
  });

  it("shows error when fields are empty", () => {
    render(<Auth />);
    fireEvent.click(screen.getByRole("button", { name: "Login" }));
    expect(screen.getByTestId("auth-error")).toHaveTextContent(
      "Username and password are required"
    );
  });

  it("shows error for invalid credentials", () => {
    render(<Auth />);
    fireEvent.change(screen.getByTestId("username-input"), {
      target: { value: "wrong" },
    });
    fireEvent.change(screen.getByTestId("password-input"), {
      target: { value: "wrong" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Login" }));
    expect(screen.getByTestId("auth-error")).toHaveTextContent(
      "Invalid credentials"
    );
  });

  it("logs in with valid credentials", () => {
    render(<Auth />);
    fireEvent.change(screen.getByTestId("username-input"), {
      target: { value: "admin" },
    });
    fireEvent.change(screen.getByTestId("password-input"), {
      target: { value: "password" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Login" }));
    expect(screen.getByText("Welcome, admin")).toBeInTheDocument();
  });

  it("logs out successfully", () => {
    render(<Auth />);
    fireEvent.change(screen.getByTestId("username-input"), {
      target: { value: "admin" },
    });
    fireEvent.change(screen.getByTestId("password-input"), {
      target: { value: "password" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Login" }));
    fireEvent.click(screen.getByRole("button", { name: "Logout" }));
    expect(screen.getByRole("heading", { name: "Login" })).toBeInTheDocument();
  });
});

