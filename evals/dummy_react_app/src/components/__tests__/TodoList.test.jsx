import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import TodoList from "../TodoList";

describe("TodoList", () => {
  it("renders the heading", () => {
    render(<TodoList />);
    expect(screen.getByText("Todo List")).toBeInTheDocument();
  });

  it("adds a todo item", () => {
    render(<TodoList />);
    const input = screen.getByTestId("todo-input");
    fireEvent.change(input, { target: { value: "Buy milk" } });
    fireEvent.click(screen.getByText("Add"));
    expect(screen.getByText("Buy milk")).toBeInTheDocument();
  });

  it("does not add an empty todo", () => {
    render(<TodoList />);
    fireEvent.click(screen.getByText("Add"));
    expect(screen.queryAllByTestId("todo-item")).toHaveLength(0);
  });

  it("toggles a todo item", () => {
    render(<TodoList />);
    const input = screen.getByTestId("todo-input");
    fireEvent.change(input, { target: { value: "Walk dog" } });
    fireEvent.click(screen.getByText("Add"));
    const item = screen.getByText("Walk dog");
    fireEvent.click(item);
    expect(item).toHaveStyle("text-decoration: line-through");
  });
});

