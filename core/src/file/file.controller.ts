import { Controller, Post, UploadedFile, UseInterceptors } from '@nestjs/common';
import { FileService } from './file.service';
import { FileInterceptor } from '@nestjs/platform-express';
import { diskStorage } from 'multer';
import { join } from 'path';
@Controller('file')
export class FileController {
  constructor(private readonly fileService: FileService) {}


  @Post('upload')
  @UseInterceptors(
    FileInterceptor('file', {
      storage: diskStorage({
        // destination: './storage',    // this both  store file in storage
        destination: join(__dirname, '..', '..', 'storage'),
        filename: (req, file, callback) => {
          const randomName =
            Date.now() + '-' + Math.round(Math.random() * 1000);
          const originalNameWithoutSpaces = file.originalname.replace(
            /\s/g,
            '_',
          );
          const fullFilename = randomName + '_' + originalNameWithoutSpaces;
          return callback(null, fullFilename);
        },
      }),
      limits: {
        fileSize: 1024 * 1024 * 5,
      },
    }),
  )
  async uploadFile(@UploadedFile() file: any) {
    return await this.fileService.uploadFile(file.filename);
  }

}
