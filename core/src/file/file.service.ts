import { HttpException, HttpStatus, Injectable } from '@nestjs/common';
import { PrismaService } from 'src/prisma/prisma.service';
import { FilesMapper } from './files.mapper';
import { randomUUID } from 'crypto';

@Injectable()
export class FileService {
  constructor(private readonly prismaService: PrismaService) {}

  async uploadFile(url: any) {
    if (!url) {
      throw new HttpException(
        'Unable to upload your file,Please try again later',
        HttpStatus.UNPROCESSABLE_ENTITY,
      );
    }
    const fileUrl = `storage/${url}`;
    const file = await this.prismaService.file.create({
      data: {
        fileUrl: fileUrl,
      },
      select: {
        id: true,
        fileUrl: true,
      },
    });
    return FilesMapper.map(file);
  }
}
