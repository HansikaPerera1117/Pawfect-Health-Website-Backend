export class FilesMapper {
  static map(file) {
    const { id, file_url } = file;
    return {
      id,
      file_url: `http://localhost:5001/${file_url}`, // Assuming file_url is the full filename
    };
  }
}
