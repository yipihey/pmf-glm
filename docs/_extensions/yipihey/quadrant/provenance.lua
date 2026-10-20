-- Julia adaptation of yipihey/quadrant's MIT-licensed source/test shortcodes.
-- Copyright (c) 2026 Tom Abel. No badge asserts a pass.
local function provenance(args, kind)
  local path = pandoc.utils.stringify(args[1] or "")
  local symbol = pandoc.utils.stringify(args[2] or "")
  local root = quarto.project.directory
  local file = io.open(root .. "/../" .. path, "r")
  if not file then error("Missing provenance file: " .. path) end
  local body = file:read("*a")
  file:close()
  local line = 1
  if symbol ~= "" then
    local position = body:find(symbol, 1, true)
    if not position then error("Missing provenance symbol " .. symbol .. " in " .. path) end
    if not symbol:find("%s") then
      local cursor = 1
      for source_line in body:gmatch("[^\n]*\n?") do
        if source_line:find(symbol, 1, true) and
           (source_line:find("function ", 1, true) or source_line:find("struct ", 1, true)) then
          position = cursor
          break
        end
        cursor = cursor + #source_line
      end
    end
    local _, count = body:sub(1, position - 1):gsub("\n", "\n")
    line = count + 1
  end
  local offset = quarto.project.offset or "."
  if offset == "" then offset = "." end
  local href = offset .. "/sources/" .. path .. ".html#L" .. line
  local label = symbol ~= "" and symbol or path:match("[^/]+$")
  local link = pandoc.Link(label, href, kind .. ": " .. path)
  if kind == "test source" then
    link.classes:insert("qd-test")
    return link
  end
  return pandoc.Superscript({link})
end
local function prepared(args)
  local name = pandoc.utils.stringify(args[1] or "")
  if name ~= "validation" and name ~= "benchmark" and name ~= "environment" and name ~= "production" then
    error("Unknown prepared document: " .. name)
  end
  local file = assert(io.open(quarto.project.directory .. "/_includes/" .. name .. ".qmd", "r"))
  local body = file:read("*a")
  file:close()
  return pandoc.read(body, "markdown").blocks
end

return {
  src = function(args) return provenance(args, "implementation") end,
  test = function(args) return provenance(args, "test source") end,
  prepared = prepared
}
