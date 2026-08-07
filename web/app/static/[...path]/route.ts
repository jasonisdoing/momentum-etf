import { readFile } from "node:fs/promises";
import path from "node:path";

import { NextResponse } from "next/server";

const STATIC_IMAGES = new Set(["HL symbol_mint green.svg"]);

export async function GET(
  _request: Request,
  context: { params: Promise<{ path: string[] }> },
) {
  const segments = (await context.params).path;
  const filename = segments.join("/");
  if (!STATIC_IMAGES.has(filename)) {
    return new NextResponse("Not Found", { status: 404 });
  }

  const filePath = path.resolve(process.cwd(), "..", "static", filename);
  const image = await readFile(filePath);
  const contentType = filename.endsWith(".svg") ? "image/svg+xml" : "image/png";
  return new NextResponse(image, {
    headers: {
      "Cache-Control": "public, max-age=86400",
      "Content-Type": contentType,
    },
  });
}
