-- Pandoc's GFM writer uses the image caption as HTML alt text.
function Image(image)
  if image.attributes["fig-alt"] then
    image.caption = {pandoc.Str(image.attributes["fig-alt"])}
    image.attributes["fig-alt"] = nil
  end
  return image
end
