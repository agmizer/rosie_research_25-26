import torch
from spire.presentation import *
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class Chunking:

    """
    This class is used to extract text from powerpoints, pdfs, and also has a chunking method
    """


    def extract_text_from_pdf(self, path_to_pdf):
        reader = PdfReader(path_to_pdf)

        pages = []

        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                pages.append(
                    Document(
                        page_content = text, 
                        metadata={
                            "source": path_to_pdf,
                            "page": i+1
                    }
                )
            )

            # this helps keep tables and figures together also right now does not keep tables intact
        return pages 

    def extract_text_from_handwritten_pdf(self, model, path_to_pdf):
        reader = PdfReader(path_to_pdf)
        num_pages = len(reader.pages)

        pages = []
        
        for i in range(num_pages):
            text = model.read_page(path_to_pdf, i)

            if text:
                pages.append(
                    Document(
                        page_content = text, 
                        metadata={
                            "source": path_to_pdf,
                            "page": i+1
                    }
                )
            )
            torch.cuda.empty_cache()
            
        return pages

        

    def extract_text_from_shape(self, shape):
        """
        Extract text from an individual shape or grouped shapes.
        """
        text = ""
        if isinstance(shape, IAutoShape):
            if shape.TextFrame is not None:
                for para in shape.TextFrame.Paragraphs:
                    text += para.Text + "\n"
        elif isinstance(shape, GroupShape):
            for sub_shape in shape.Shapes:
                text += self.extract_text_from_shape(sub_shape)
        return text

    
    def extract_all_content(self, ppt_path):
        """
        Extract all text content from shapes, tables, smartart, and speaker notes
        from a PowerPoint file.

        Returns a list of Documents containing all extracted information separated by page.
        """

        presentation = Presentation()
        presentation.LoadFromFile(ppt_path)

        docs = []

        for slide_index, slide in enumerate(presentation.Slides):

            slide_content = f"--- Slide {slide_index + 1} ---\n"

            shape_text = ""
            table_text = ""
            smartart_text = ""
            notes_text = ""

        
            for shape in slide.Shapes:
                # tables
                if isinstance(shape, ITable):
                    for row in shape.TableRows:
                        for cell in row:
                            if cell.TextFrame is not None:
                                for para in cell.TextFrame.Paragraphs:
                                    table_text += para.Text + "\t"
                        table_text += "\n"
                # smartart
                elif isinstance(shape, ISmartArt):
                    for node in shape.Nodes:
                        if node.TextFrame is not None:
                            smartart_text += node.TextFrame.Text + "\n"
                # textboxes
                else:
                    shape_text += self.extract_text_from_shape(shape)

            # speaker notes
            notes_slide = slide.NotesSlide
            if notes_slide is not None and notes_slide.NotesTextFrame is not None:
                notes_text = notes_slide.NotesTextFrame.Text

            # Combine sections (1 newline between extraction methods)
            slide_content += shape_text.strip() + "\n"
            slide_content += table_text.strip() + "\n"
            slide_content += smartart_text.strip() + "\n"
            slide_content += notes_text.strip()

            docs.append(
                Document(
                    page_content = slide_content.strip(),
                    metadata={
                        "source": ppt_path,
                        "page": slide_index + 1
                    }
                )
            )

        presentation.Dispose()
        return docs



    @staticmethod
    def chunk_with_langchain(docs, source_name="unknown"):
        """
        Convert raw text into LangChain documents and chunk them.
        """

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )
        chunks = splitter.split_documents(docs)
        return chunks

def main():
    chunker = Chunking()
    #print(chunker.extract_text_from_pdf("pdfs/Abstract Algebra Syllabus Spring 2026.pdf"))
    #print(chunker.extract_all_content("pdfs/Week1D1_Intro-1.pptx"))
    print(Chunking.chunk_with_langchain(chunker.extract_text_from_pdf("pdfs/Abstract Algebra Syllabus Spring 2026.pdf"), "Abstract Algebra Syllabus 2026"))

if __name__ == "__main__":
    main()